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

// ── Domain classification ──────────────────────────────────────────────────

static DOMAIN_PATTERNS: Lazy<Vec<(&'static str, Vec<&'static str>)>> = Lazy::new(|| {
    vec![
        ("rust",       vec!["rust", "cargo", "crate", "tokio", "async fn", "impl ", "trait ", "borrow", "lifetime", "rustc", ".rs"]),
        ("python",     vec!["python", "pip", "django", "flask", "pandas", "numpy", "pytorch", "tensorflow", "def ", ".py", "import ", "__init__"]),
        ("javascript", vec!["javascript", "nodejs", "node.js", "npm", "webpack", "react", "vue", "angular", "const ", "let ", "var ", ".js", "console.log"]),
        ("typescript", vec!["typescript", "tsx", ".ts", "interface ", "type ", "tsc", "deno"]),
        ("sql",        vec!["select ", "insert ", "update ", "delete ", "create table", "alter table", "drop table", "join ", "where ", "sql", "postgres", "mysql", "sqlite"]),
        ("shell",      vec!["bash", "#!/bin/sh", "#!/bin/bash", "shell", "zsh", "grep ", "awk ", "sed ", "chmod", "curl ", "wget", "systemctl", "apt ", "brew "]),
        ("go",         vec!["golang", "go mod", "func ", "package main", " go ", ".go"]),
        ("c",          vec!["#include <", "printf(", "malloc(", "free(", "int main", ".c ", ".h "]),
        ("cpp",        vec!["#include <iostream>", "std::", "cout <<", "nullptr", "template<", ".cpp", ".hpp"]),
        ("java",       vec!["java", "public class", "import java.", "maven", "gradle", ".java"]),
        ("assembly",   vec!["mov ", "push ", "pop ", "jmp ", "call ", "ret ", "nasm", "gas ", ".asm", "register"]),
        ("docker",     vec!["dockerfile", "docker-compose", "container", "image", "kubernetes", "k8s", "helm"]),
        ("git",        vec!["git ", "github", "gitlab", "commit", "branch", "merge", "rebase", "pull request"]),
        ("toml",       vec!["toml", ".toml", "[dependencies]", "[package]"]),
        ("yaml",       vec!["yaml", ".yml", ".yaml", "kind: ", "apiVersion:"]),
    ]
});

static INTENT_PATTERNS: Lazy<Vec<(&'static str, Vec<&'static str>)>> = Lazy::new(|| {
    vec![
        ("fix",       vec!["fix", "bug", "error", "broken", "crash", "doesn't work", "not working", "wrong", "failed", "failure", "issue", "problem", "debug"]),
        ("explain",   vec!["explain", "what is", "what are", "how does", "why does", "describe", "tell me", "clarify", "difference between", "meaning of"]),
        ("generate",  vec!["generate", "write", "create", "implement", "build", "make", "add", "scaffold", "template"]),
        ("review",    vec!["review", "check", "audit", "look at", "feedback", "improve", "critique", "analyze", "best practice"]),
        ("refactor",  vec!["refactor", "clean up", "simplify", "restructure", "reorganize", "extract", "rename"]),
        ("optimize",  vec!["optimize", "performance", "faster", "slower", "bottleneck", "profile", "speed up", "memory usage"]),
        ("summarize", vec!["summarize", "summary", "brief", "overview", "tldr", "digest"]),
        ("convert",   vec!["convert", "translate", "migrate", "port", "transform", "rewrite in"]),
        ("test",      vec!["test", "unit test", "integration test", "spec", "assert", "mock", "coverage"]),
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
    m.insert("review",   0.70);
    m.insert("assembly", 0.60);
    m.insert("c",        0.60);
    m.insert("cpp",      0.60);
    m.insert("rust",     0.40);
    m.insert("python",   0.40);
    m.insert("typescript", 0.40);
    m.insert("shell",    0.35);
    m.insert("javascript", 0.35);
    m.insert("sql",      0.25);
    m.insert("toml",     0.20);
    m.insert("yaml",     0.20);
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
    let mut best = ("general", 0usize);
    for (name, keywords) in DOMAIN_PATTERNS.iter() {
        let count = keywords.iter().filter(|k| lower.contains(*k)).count();
        if count > best.1 {
            best = (name, count);
        }
    }
    best.0.to_string()
}

pub fn classify_intent(lower: &str) -> String {
    let mut best = ("general", 0usize);
    for (name, keywords) in INTENT_PATTERNS.iter() {
        let count = keywords.iter().filter(|k| lower.contains(*k)).count();
        if count > best.1 {
            best = (name, count);
        }
    }
    best.0.to_string()
}

pub fn classify_complexity(lower: &str, domain: &str, intent: &str) -> f64 {
    let mut score = base_complexity(domain, intent);

    let boost = COMPLEXITY_BOOSTERS.iter().filter(|k| lower.contains(*k)).count();
    let reduce = COMPLEXITY_REDUCERS.iter().filter(|k| lower.contains(*k)).count();

    // Each booster adds ~0.06, each reducer subtracts ~0.06
    score += boost as f64 * 0.06;
    score -= reduce as f64 * 0.06;

    // Long prompts are more complex
    let word_count = lower.split_whitespace().count();
    if word_count > 200 { score += 0.10; }
    else if word_count > 100 { score += 0.05; }

    score.clamp(0.0, 1.0)
}

fn base_complexity(domain: &str, intent: &str) -> f64 {
    let domain_base = match domain {
        "assembly" => 0.70,
        "c" | "cpp" => 0.55,
        "rust" => 0.50,
        "python" | "java" | "go" => 0.40,
        "javascript" | "typescript" => 0.38,
        "sql" => 0.30,
        "shell" => 0.28,
        "docker" | "yaml" | "toml" => 0.20,
        _ => 0.30,
    };
    let intent_mod = match intent {
        "generate" | "implement" => 0.10,
        "review" | "refactor" => 0.08,
        "optimize" => 0.06,
        "convert" | "test" => 0.04,
        "fix" => 0.02,
        "explain" | "summarize" => -0.05,
        _ => 0.0,
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
