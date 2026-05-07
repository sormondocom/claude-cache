use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

/// A classified prompt shape used as the semantic cache scope and routing input.
#[derive(Debug, Clone)]
pub struct ShapeKey {
    pub domain:     String,
    pub intent:     String,
    pub complexity: f64,
}

impl ShapeKey {
    pub fn display(&self) -> String {
        format!("{}:{}:{:.2}", self.domain, self.intent, self.complexity)
    }
}

// ── Domain classification ─────────────────────────────────────────────────
//
// Each entry is (keyword, weight).  Weights are additive; the domain with the
// highest total score wins.  Higher weights (3.0-5.0) are reserved for
// keywords that are diagnostic / unique to that language.  Lower weights
// (0.5-1.5) cover terms that appear across multiple languages so they
// contribute without dominating.
//
// Key design choices:
//  - TypeScript has strong unique signals (": string", "interface ", etc.) so
//    it beats JavaScript even when generic JS keywords like "const " appear.
//  - "const " and "let " have low weight in JS (0.5) since they are also
//    valid Rust syntax and appear in TS/Go/etc.
//  - Languages are checked in declaration order; ties go to the first match.

static DOMAIN_PATTERNS: Lazy<Vec<(&'static str, Vec<(&'static str, f64)>)>> = Lazy::new(|| {
    vec![
        ("rust", vec![
            ("async fn",  4.0), ("pub fn",     4.0), ("impl ",     3.5),
            ("#[derive",  4.0), ("&mut ",       3.0), ("fn main(",  3.5),
            ("rustc",     4.0), ("tokio",       3.0), ("trait ",    2.5),
            ("Box<",      2.5), ("Arc<",        2.5), ("Vec<",      2.0),
            ("Option<",   2.0), ("Result<",     2.0), ("cargo",     2.0),
            ("crate",     2.0), ("borrow",      1.5), ("lifetime",  1.5),
            (".rs",       2.0), ("rust",        1.0), ("mod ",      1.5),
            ("use std::", 3.0),
        ]),
        ("typescript", vec![
            (": string",  4.0), (": number",    4.0), (": boolean", 4.0),
            ("interface", 4.0), ("readonly ",   3.5), ("as const",  3.5),
            (".tsx",      4.0), ("tsx",          2.5), (".ts",       3.0),
            ("typescript",3.0), ("tsc",          2.5), ("deno",      2.5),
            ("type alias",2.5), ("type guard",  3.0), ("satisfies ",3.5),
            ("infer ",    3.0), ("keyof ",       3.5), ("Partial<",  2.5),
            ("Record<",   2.5), ("Awaited<",    3.0),
        ]),
        ("javascript", vec![
            ("console.log", 4.0), ("require(",  4.0), ("module.exports", 4.0),
            ("nodejs",    3.5), ("node.js",     3.5), ("npm ",      2.5),
            ("webpack",   3.0), ("react",        2.5), ("vue",       2.5),
            ("angular",   2.5), ("javascript",  3.0), (".js",       2.0),
            ("async/await",2.0), ("promise",   2.0), ("callback",  2.0),
            ("const ",    0.5), ("let ",        0.5), ("var ",      1.5),
        ]),
        ("python", vec![
            ("def ",      3.5), ("self.",        3.5), ("__init__",  4.0),
            ("pip install",4.0), ("pip ",        2.5), ("print(",    2.0),
            ("python",    2.5), (".py",           2.5), ("django",    2.5),
            ("flask",     2.5), ("pandas",       3.0), ("numpy",     3.0),
            ("pytorch",   3.0), ("tensorflow",  3.0), ("import numpy",4.0),
            ("import pandas",4.0), ("import torch",4.0), ("lambda ", 2.0),
            ("import ",   1.0), ("elif ",        2.5), ("__name__",  3.0),
        ]),
        ("sql", vec![
            ("select ",   2.0), ("insert into", 4.0), ("delete from",4.0),
            ("create table",4.0), ("alter table",4.0), ("drop table",4.0),
            ("inner join",4.0), ("left join",   4.0), ("right join",4.0),
            ("join ",     2.0), ("where ",       1.5), ("group by",  3.5),
            ("order by",  3.0), ("having ",      3.5), ("sql",       2.5),
            ("postgres",  3.0), ("mysql",        3.0), ("sqlite",    3.0),
            ("from ",     1.0),
        ]),
        ("shell", vec![
            ("#!/bin/",   5.0), ("bash",         3.0), ("zsh",       3.0),
            ("shell script",4.0), ("grep ",      2.5), ("awk ",      3.0),
            ("sed ",      2.5), ("chmod",         3.5), ("curl ",    2.5),
            ("wget",      3.0), ("systemctl",    4.0), ("apt ",      3.5),
            ("brew ",     3.0), ("$(", 2.0),           ("| grep",   2.5),
            ("export ",   1.5),
        ]),
        ("go", vec![
            ("package main",4.0), ("func ",     3.5), ("go mod",    4.0),
            ("golang",    4.0), ("goroutine",   4.0), ("chan ",      3.5),
            (":=",        2.5), (".go",           3.0), ("go.sum",   4.0),
            ("fmt.Print", 4.0), ("go get",      4.0),
        ]),
        ("c", vec![
            ("int main(", 4.0), ("#include <stdio.h>",4.0), ("#include <stdlib.h>",4.0),
            ("printf(",   3.0), ("malloc(",     3.5), ("free(",     3.0),
            ("sizeof(",   3.0), ("typedef ",    2.5), ("NULL",      2.0),
            ("#define ",  2.5),
        ]),
        ("cpp", vec![
            ("#include <iostream>",5.0), ("std::", 4.0), ("cout <<",  4.0),
            ("nullptr",   4.0), ("template<",   4.0), ("virtual ",  3.0),
            ("override",  2.5), (".cpp",         3.5), (".hpp",      3.5),
            ("unique_ptr",4.0), ("shared_ptr",  4.0),
        ]),
        ("java", vec![
            ("public class",4.0), ("import java.",5.0), ("System.out",4.0),
            ("@Override", 4.0), ("throws ",     3.0), ("new ArrayList",3.5),
            ("maven",     3.5), ("gradle",       3.5), (".java",     3.0),
            ("java",      1.5), ("@SpringBoot", 4.0),
        ]),
        ("assembly", vec![
            ("nasm",      5.0), (" eax",         3.5), (" ebp",     3.5),
            (" esp,",     3.5), ("jmp ",          3.0), (".asm",    4.0),
            ("assembly",  3.0), (".section",     3.5), (".global",  3.5),
        ]),
        ("docker", vec![
            ("dockerfile",4.0), ("FROM ",        3.5), ("RUN ",     2.5),
            ("docker-compose",4.0), ("kubernetes",4.0), ("k8s",     4.0),
            ("helm",      3.0), ("container",    2.0), ("docker build",4.0),
        ]),
        ("git", vec![
            ("git commit",4.0), ("git push",    4.0), ("git pull",  4.0),
            ("git merge", 4.0), ("git rebase",  4.0), ("pull request",4.0),
            ("github",    3.0), ("gitlab",       3.0), ("git branch",3.5),
            ("git log",   3.5), ("git status",  3.5),
        ]),
        ("toml", vec![
            ("[dependencies]",5.0), ("[package]",4.0), ("toml",    3.0),
            (".toml",     3.5),
        ]),
        ("yaml", vec![
            ("apiversion:",5.0), ("kind: ",     4.0), ("yaml",      3.0),
            (".yml",      3.0), (".yaml",        3.0),
        ]),
    ]
});

// ── Intent classification ─────────────────────────────────────────────────
//
// Same weighted approach.  "add" and "make" have low weight in "generate"
// because they appear in too many other contexts.

static INTENT_PATTERNS: Lazy<Vec<(&'static str, Vec<(&'static str, f64)>)>> = Lazy::new(|| {
    vec![
        ("fix", vec![
            ("fix",         3.5), ("bug",         3.0), ("error",    2.5),
            ("broken",      3.0), ("crash",        3.0), ("debug",   2.5),
            ("fail",        2.5), ("not working",  3.5), ("doesn't work",3.5),
            ("wrong",       1.5), ("issue",        1.5), ("problem", 1.5),
        ]),
        ("explain", vec![
            ("explain",     4.0), ("what is",      3.0), ("what are",3.0),
            ("how does",    3.0), ("why does",     3.0), ("describe",2.5),
            ("tell me",     2.0), ("clarify",      3.0), ("difference between",3.5),
            ("meaning of",  3.0), ("what does",    3.0), ("how do",  1.5),
        ]),
        ("generate", vec![
            ("generate",    3.5), ("write",        3.0), ("create",  3.0),
            ("implement",   3.5), ("build",        3.0), ("scaffold",4.0),
            ("template",    3.0), ("make",         1.0), ("add",     0.5),
        ]),
        ("review", vec![
            ("review",      4.0), ("audit",        4.0), ("critique",4.0),
            ("feedback",    3.0), ("improve",      2.5), ("analyze", 3.0),
            ("best practice",4.0), ("look at",     2.0), ("check",  2.5),
        ]),
        ("refactor", vec![
            ("refactor",    5.0), ("clean up",     4.0), ("simplify",3.5),
            ("restructure", 4.0), ("reorganize",   4.0), ("extract", 3.0),
            ("rename",      3.5),
        ]),
        ("optimize", vec![
            ("optimize",    5.0), ("performance",  4.0), ("faster",  3.0),
            ("bottleneck",  4.0), ("profile",      3.5), ("speed up",4.0),
            ("memory usage",3.5), ("slower",       2.0),
        ]),
        ("summarize", vec![
            ("summarize",   5.0), ("summary",      4.0), ("overview",3.0),
            ("tldr",        5.0), ("digest",        3.0), ("brief",  2.5),
        ]),
        ("convert", vec![
            ("convert",     5.0), ("translate",    4.0), ("migrate", 4.0),
            ("transform",   3.5), ("rewrite in",   5.0), ("port",   2.5),
        ]),
        ("test", vec![
            ("unit test",   5.0), ("integration test",5.0), ("coverage",4.0),
            ("mock",        3.5), ("assert",       3.5), ("spec",    3.5),
            ("test",        2.5),
        ]),
    ]
});

// Patterns that bump complexity up
static COMPLEXITY_BOOSTERS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "architecture", "distributed", "concurrent", "parallel", "async", "lifetime",
        "generic", "macro", "unsafe", "algorithm", "optimization", "security", "auth",
        "encryption", "cryptography", "consensus", "transaction", "sharding", "replication",
        "implement from scratch", "design pattern", "abstract", "polymorphism",
    ]
});

static COMPLEXITY_REDUCERS: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "hello world", "simple", "basic", "beginner", "quick", "easy", "just", "only",
        "snippet", "example", "demo", "tutorial", "how to print", "rename",
    ]
});

// Consequence overrides — these domains/intents carry inherent risk
static CONSEQUENCE_MAP: Lazy<HashMap<&'static str, f64>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("review",      0.70);
    m.insert("assembly",    0.60);
    m.insert("c",           0.60);
    m.insert("cpp",         0.60);
    m.insert("rust",        0.40);
    m.insert("python",      0.40);
    m.insert("typescript",  0.40);
    m.insert("shell",       0.35);
    m.insert("javascript",  0.35);
    m.insert("sql",         0.25);
    m.insert("toml",        0.20);
    m.insert("yaml",        0.20);
    m
});

static VERSION_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(v?\d+\.\d+[\.\d]*|latest|nightly|stable|beta|alpha|rc\d*)\b").unwrap()
});

static RECENCY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(new|latest|recent|2024|2025|2026|updated|released|just|current|today)\b").unwrap()
});

pub fn classify(prompt: &str) -> ShapeKey {
    let lower = prompt.to_lowercase();

    let domain = classify_domain(&lower);
    let intent = classify_intent(&lower);
    let complexity = classify_complexity(&lower, &domain, &intent);

    ShapeKey { domain, intent, complexity }
}

pub fn classify_domain(lower: &str) -> String {
    let mut best = ("general", 0.0f64);
    for (name, keywords) in DOMAIN_PATTERNS.iter() {
        let score: f64 = keywords.iter()
            .filter(|(k, _)| lower.contains(*k))
            .map(|(_, w)| w)
            .sum();
        if score > best.1 {
            best = (name, score);
        }
    }
    best.0.to_string()
}

pub fn classify_intent(lower: &str) -> String {
    let mut best = ("general", 0.0f64);
    for (name, keywords) in INTENT_PATTERNS.iter() {
        let score: f64 = keywords.iter()
            .filter(|(k, _)| lower.contains(*k))
            .map(|(_, w)| w)
            .sum();
        if score > best.1 {
            best = (name, score);
        }
    }
    best.0.to_string()
}

pub fn classify_complexity(lower: &str, domain: &str, intent: &str) -> f64 {
    let mut score = base_complexity(domain, intent);

    let boost  = COMPLEXITY_BOOSTERS.iter().filter(|k| lower.contains(*k)).count();
    let reduce = COMPLEXITY_REDUCERS.iter().filter(|k| lower.contains(*k)).count();

    score += boost  as f64 * 0.06;
    score -= reduce as f64 * 0.06;

    let word_count = lower.split_whitespace().count();
    if word_count > 200      { score += 0.10; }
    else if word_count > 100 { score += 0.05; }

    score.clamp(0.0, 1.0)
}

fn base_complexity(domain: &str, intent: &str) -> f64 {
    let domain_base = match domain {
        "assembly"              => 0.70,
        "c" | "cpp"             => 0.55,
        "rust"                  => 0.50,
        "python" | "java" | "go"=> 0.40,
        "javascript" | "typescript" => 0.38,
        "sql"                   => 0.30,
        "shell"                 => 0.28,
        "docker" | "yaml" | "toml" => 0.20,
        _                       => 0.30,
    };
    let intent_mod = match intent {
        "generate" | "implement" => 0.10,
        "review"   | "refactor"  => 0.08,
        "optimize"               => 0.06,
        "convert"  | "test"      => 0.04,
        "fix"                    => 0.02,
        "explain"  | "summarize" => -0.05,
        _                        => 0.0,
    };
    domain_base + intent_mod
}

pub fn consequence_score(domain: &str, intent: &str) -> f64 {
    let by_intent = CONSEQUENCE_MAP.get(intent).copied().unwrap_or(0.0);
    let by_domain = CONSEQUENCE_MAP.get(domain).copied().unwrap_or(0.20);
    by_intent.max(by_domain)
}

pub fn has_version_specifier(prompt: &str) -> bool {
    VERSION_RE.is_match(prompt)
}

pub fn has_recency_signal(prompt: &str) -> bool {
    RECENCY_RE.is_match(&prompt.to_lowercase())
}
