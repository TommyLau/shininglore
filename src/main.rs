mod bwx;

fn main() {
    println!("Hello, world!");
    let mut b = bwx::BWX::new();
    let c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX");
    match c {
        Ok(()) => eprintln!("OK!"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
