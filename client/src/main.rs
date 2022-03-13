use bwx::BWX;
use tracing::Level;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(if cfg!(debug_assertions) { Level::DEBUG } else { Level::INFO })
        .init();

    let mut b = BWX::new();
    let c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX");
    //let c = b.load_from_file("MON045_DEADA.PNX");
    match c {
        Ok(()) => eprintln!("OK!"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
