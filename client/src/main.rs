use bwx::BWX;
use tracing::{debug, Level};
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
    let _c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX")?;
    //let c = b.load_from_file("MON045_DEADA.PNX");
    //let _c = b.load_from_file("MON039_DEFAULTA.PNX")?;
    //let _c = b.load_from_file("MON006_DEFAULTA.PNX");
    //let _c = b.load_from_file("MON001_01_DEFAULTA.PNX")?;
    //let _c = b.load_from_file("OBO020_DEFAULT.PNX")?;


    let b: Vec<u8> = vec![
        250, 130, 198, 193, 245, 168, 13, 194, 190, 58, 156, 192,
        206, 82, 109, 190, 38, 227, 202, 188, 248, 242, 120, 63,
        141, 31, 190, 62, 144, 215, 187, 187,
    ];
    let p = b.as_ptr() as *const f32;

    for i in 0..(b.len() / 4) as isize {
        debug!("{:?}", unsafe {*p.offset(i)});
        let j = 4 * i as usize;
        let f = f32::from_le_bytes(b[j..j + 4].try_into().unwrap());
        debug!("{:?}", f);
    }

    Ok(())
}
