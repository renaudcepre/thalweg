use std::process::Command;

fn main() {
    let hash = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());

    // --untracked-files=no: untracked files (scratch, worktrees, personal notes)
    // don't count as "dirty", want to flag only uncommitted changes to versioned code.
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .is_ok_and(|o| !o.stdout.is_empty());

    let hash_tag = if dirty { format!("{hash}-dirty") } else { hash };

    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    println!("cargo:rustc-env=HEXSIM_BUILD_HASH={hash_tag}");
    println!("cargo:rustc-env=HEXSIM_BUILD_UNIX={unix}");
    // Paths relative to CARGO_MANIFEST_DIR (= crates/hexsim-cli/)
    // ascend to repo root (simulation/../.git)
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/index");
}
