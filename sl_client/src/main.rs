use clap::Parser;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::filter::EnvFilter;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// User name to generate license
    filename: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(filter) = EnvFilter::try_from_default_env() {
        if cfg!(debug_assertions) {
            tracing_subscriber::fmt().with_env_filter(filter).init();
        } else {
            match filter.to_string().as_str() {
                "error" | "warn" | "info" => {
                    tracing_subscriber::fmt().with_env_filter(filter).init();
                }
                _ => {}
            }
        }
    } else {
        tracing_subscriber::fmt()
            .with_max_level(if cfg!(debug_assertions) {
                Level::DEBUG
            } else {
                Level::ERROR
            })
            .init();
    }

    // Parse command line
    let cli = Cli::parse();

    let filename = if let Some(f) = cli.filename {
        f
    } else {
        "Assets/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_DEFAULT.PNX".into()
    };

    let mut gltf = sl::Gltf::new();
    // gltf.load_from_bwx("Assets/Graphic/MONSTER/MON059.PNX/MON059_DEFAULTA.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/INTERFACE/Login_stage.pnx/LOGIN_STAGE_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROETO.PNX/HEROETO_DEFAULT.PNX")?;
    // gltf.load_from_bwx("../sl2_data/Graphic/NPC/WORLD01/HEROETO.PNX/HEROETO_DEFAULT.PNX")?;
    // gltf.load_from_bwx("../sl2_data/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_WALK.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/NPC_CHINAM002ghost.PNX/NPC_CHINAFM002GHOST_DEFAULT.PNX")?;

    // NPC Hero
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROBIO.PNX/HEROBIO_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROETO.PNX/HEROETO_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROMENE.PNX/HEROMENE_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HERORUNE.PNX/HERORUNE_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROSERINE.PNX/HEROSERINE_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROSIENA.PNX/HEROSIENA_DEFAULT.PNX")?;
    // gltf.load_from_bwx("Assets/Graphic/NPC/WORLD01/HEROWIN.PNX/HEROWIN_DEFAULT.PNX")?;
    gltf.load_from_bwx(filename)?;
    gltf.save_gltf("./tmp2")?;

    Ok(())
}
