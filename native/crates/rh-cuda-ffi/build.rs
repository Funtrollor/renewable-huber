use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=RH_CUDA_LINK_KIND");
    println!("cargo:rerun-if-env-changed=RH_CUDA_CMAKE_PROFILE");
    println!("cargo:rerun-if-env-changed=RH_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-changed=../../cuda/CMakeLists.txt");
    println!("cargo:rerun-if-changed=../../cuda/include/rh_cuda.h");
    // Cargo watches a directory recursively, so CUDA kernel/header edits cannot
    // accidentally reuse a stale CMake archive from a previous build.
    println!("cargo:rerun-if-changed=../../cuda/src");

    // A normal CPU/source installation must not require nvcc or a CUDA driver.
    // The C ABI is compiled and linked only for the explicit `cuda` feature.
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets it"));
    let cuda_dir = manifest_dir.join("../../cuda");
    let cmake_lists = cuda_dir.join("CMakeLists.txt");
    let header = cuda_dir.join("include/rh_cuda.h");
    if !cmake_lists.is_file() || !header.is_file() {
        panic!(
            "the `cuda` feature requires native/cuda/CMakeLists.txt and native/cuda/include/rh_cuda.h; \
             build without `--features cuda` on CPU-only hosts"
        );
    }

    let profile = env::var("RH_CUDA_CMAKE_PROFILE").unwrap_or_else(|_| "Release".to_owned());
    // Source builds must generate code for the developer's active GPU unless
    // the wheel/release build supplies an explicit fat-binary architecture
    // list (for example `90;120`).  This host is SM 12.0, so nvcc's implicit
    // default would be unsafe for P2 development.
    let architectures = env::var("RH_CUDA_ARCHITECTURES").unwrap_or_else(|_| "native".to_owned());
    let destination = cmake::Config::new(&cuda_dir)
        .profile(&profile)
        // Link the project static archive into the extension.  Bundling a
        // separate rh_cuda DLL next to a Maturin-produced .pyd would require
        // platform-specific wheel repair; CUDA runtime libraries remain
        // dynamic dependencies supplied by the installed Toolkit/driver.
        .define("RH_CUDA_BUILD_SHARED", "OFF")
        .define("RH_CUDA_BUILD_TESTS", "OFF")
        .define("CMAKE_CUDA_ARCHITECTURES", architectures)
        .build();

    // `cmake::Config::build` runs the install target, placing the archive in
    // destination/lib where Cargo can link it reliably.  A release build may
    // override the linkage only for an intentionally packaged variant.
    let link_kind = env::var("RH_CUDA_LINK_KIND").unwrap_or_else(|_| "static".to_owned());
    match link_kind.as_str() {
        "dylib" | "static" => {}
        _ => panic!("RH_CUDA_LINK_KIND must be `dylib` or `static`"),
    }
    println!(
        "cargo:rustc-link-search=native={}",
        destination.join("lib").display()
    );
    println!("cargo:rustc-link-lib={link_kind}=renewable_huber_cuda");

    // The CMake static archive deliberately leaves CUDA runtime libraries
    // dynamic.  Cargo does not consume CMake target usage requirements, so
    // name the transitive libraries explicitly for the extension linker.
    if let Some(library_dir) = cuda_library_dir() {
        println!("cargo:rustc-link-search=native={}", library_dir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusolver");

    // The engine is C++ and throws, so the archive needs the C++ runtime for
    // its exception personality routine. MSVC supplies this through default
    // library directives embedded in the .lib, which is why linking worked on
    // Windows; rustc passes -nodefaultlibs, so with the GNU toolchain the
    // extension links cleanly and then fails at import with
    // `undefined symbol: __gxx_personality_v0`.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}

/// Locate the directory holding the CUDA runtime libraries.
///
/// The Windows installer exports `CUDA_PATH`, so relying on it alone worked
/// there and silently produced `unable to find library -lcudart` on Linux,
/// where no installer sets it.  Search the usual environment variables, then
/// derive the prefix from whichever `nvcc` is on `PATH`, then fall back to the
/// conventional location.
///
/// Returning `None` is not a failure: a distribution-packaged toolkit installs
/// into a directory the linker already searches.
fn cuda_library_dir() -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
        .iter()
        .filter_map(env::var_os)
        .map(PathBuf::from)
        .collect();

    // nvcc lives in <root>/bin, so its grandparent is the toolkit root. This is
    // the most reliable signal available: the CUDA sources have just been
    // compiled, so some nvcc was certainly found.
    if let Some(path) = env::var_os("PATH") {
        for directory in env::split_paths(&path) {
            let nvcc = directory.join(if cfg!(target_os = "windows") {
                "nvcc.exe"
            } else {
                "nvcc"
            });
            if nvcc.is_file() {
                if let Some(root) = directory.parent() {
                    roots.push(root.to_path_buf());
                }
            }
        }
    }

    roots.push(PathBuf::from(if cfg!(target_os = "windows") {
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    } else {
        "/usr/local/cuda"
    }));

    // `lib64` is a symlink to `targets/<arch>/lib` on a standard Linux install;
    // both are listed so an unusual layout still resolves.
    let suffixes: &[&str] = if cfg!(target_os = "windows") {
        &["lib/x64"]
    } else {
        &["lib64", "targets/x86_64-linux/lib", "lib"]
    };

    roots.iter().find_map(|root| {
        suffixes
            .iter()
            .map(|suffix| root.join(suffix))
            .find(|candidate| candidate.is_dir())
    })
}
