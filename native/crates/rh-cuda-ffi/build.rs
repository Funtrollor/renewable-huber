use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=RH_CUDA_LINK_KIND");
    println!("cargo:rerun-if-env-changed=RH_CUDA_CMAKE_PROFILE");
    println!("cargo:rerun-if-env-changed=RH_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
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
    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        let cuda_path = PathBuf::from(cuda_path);
        let library_dir = if cfg!(target_os = "windows") {
            cuda_path.join("lib/x64")
        } else {
            cuda_path.join("lib64")
        };
        if library_dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", library_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cusolver");
}
