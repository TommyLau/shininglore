use bwx::BWX;
use tracing::{error, Level, warn};
use tracing_subscriber::{FmtSubscriber};

fn main() {
    #[cfg(debug_assertions)]
    {
        // a builder for `FmtSubscriber`.
        let subscriber = FmtSubscriber::builder()
            // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
            // will be written to stdout.
            .with_max_level(Level::TRACE)
            // completes the builder.
            .finish();
        tracing::subscriber::set_global_default(subscriber)
            .expect("setting default subscriber failed");
    }

    warn!("Hello, world!");
    error!("Hello, world!");
    let mut b = BWX::new();
    let c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX");
    match c {
        Ok(()) => eprintln!("OK!"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
