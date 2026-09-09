use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=openings.js");

    let raw =
        fs::read_to_string("openings.js").expect("failed to read openings.js");

    // Strip line comments.
    let no_comments: String = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");

    // Find the JSON object.
    let start = no_comments
        .find('{')
        .expect("openings.js: could not find opening '{'");

    let end = no_comments
        .rfind('}')
        .expect("openings.js: could not find closing '}'");

    let json_body = &no_comments[start..=end];

    let book: BTreeMap<String, BTreeMap<String, u32>> =
        serde_json::from_str(json_body).expect(
            "openings.js: body is not valid JSON",
        );

    let out_dir = env::var("OUT_DIR")
        .expect("OUT_DIR environment variable missing");

    let dest_path = Path::new(&out_dir).join("opening_book_data.rs");

    let mut slice_defs = String::new();
    let mut phf_map = phf_codegen::Map::new();

    let mut slice_names: Vec<(String, String)> = Vec::new();

    for (index, (fen_key, replies)) in book.iter().enumerate() {
        let const_name = format!("BOOK_SLICE_{index}");

        let mut entries = String::new();

        for (uci, weight) in replies {
            entries.push_str(&format!(
                "(\"{}\", {}u32), ",
                escape(uci),
                weight
            ));
        }

        slice_defs.push_str(&format!(
            "#[allow(unused_variables, dead_code, unused_imports, unused_macros)]\nstatic {const_name}: &[(&str, u32)] = &[{entries}];\n"
        ));

        slice_names.push((fen_key.clone(), const_name));
    }

    for (fen_key, const_name) in &slice_names {
        phf_map.entry(fen_key.clone(), const_name);
    }

    #[allow(unused_variables)]
    let phf_src = format!(
        r#"{slice_defs}

static OPENING_BOOK: ::phf::Map<
    &'static str,
    &'static [(&'static str, u32)]
> = {};
"#,
        phf_map.build()
    );

    #[allow(unused_variables)]
    let phf_src = format!(
        "{slice_defs}\n\n#[allow(unused_variables, dead_code, unused_imports, unused_macros)]\nstatic OPENING_BOOK: ::phf::Map<\n    &'static str,\n    &'static [(&'static str, u32)]\n> = {};\n",
        phf_map.build()
    );

    fs::write(&dest_path, phf_src)
        .expect("failed to write generated opening_book_data.rs");
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
}
