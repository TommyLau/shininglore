use bwx::BWX;
use tracing::Level;
use tracing_subscriber::filter::EnvFilter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        if cfg!(debug_assertions) {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        } else {
            match filter.to_string().as_str() {
                "error" | "warn" | "info" => { tracing_subscriber::fmt().with_env_filter(filter).init(); }
                _ => {}
            }
        }
    } else {
        tracing_subscriber::fmt()
            .with_max_level(if cfg!(debug_assertions) { Level::DEBUG } else { Level::ERROR })
            .init();
    }

    let mut b = BWX::new();
    // let _c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX")?;
    //let _c = b.load_from_file("MON045_DEADA.PNX")?;
    // let _c = b.load_from_file("MON039_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("MON006_DEFAULTA.PNX");
    // let _c = b.load_from_file("MON001_01_DEFAULTA.PNX")?;
    //let _c = b.load_from_file("OBO020_DEFAULT.PNX")?;
    //let _c = b.load_from_file("MON052_DEFAULTA.PNX")?;
    let _c = b.load_from_file("MON083_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("HEROSANDRA_DEFAULT.PNX")?;
    // let _c = b.load_from_file("HEROETO_DEFAULT.PNX")?;

    Ok(())
}
