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

    let mut gltf = sl::Gltf::new();
    gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_WALK.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/MONSTER/MON059.PNX/MON059_DEFAULTA.PNX")?;
    gltf.save_gltf()?;

    Ok(())
}
