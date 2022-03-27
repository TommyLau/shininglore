use gltf::json::validation::Checked::Valid;
use bwx::BWX;
use tracing::{debug, Level};
use tracing::field::debug;
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


    // Test code for glTF round trip?

    use std::fs;
    use std::io;
    let path = "MON001_01.gltf";

    /*
    let file = fs::File::open(&path)?;
    let reader = io::BufReader::new(file);
    let gltf = gltf::Gltf::from_reader(reader)?;

     */

    /*
    let gltf = gltf::Gltf::open("MON001_01.gltf")?;
    //Vec::new();

    debug!("{:#?}", gltf);

    return Ok(());
    debug!("{:#?}", gltf.document);
    let json = gltf.document.into_json().to_string_pretty()?;
    debug!("data: {:#?}", gltf.blob);
    let data = if gltf.blob.is_some() { gltf.blob.unwrap() } else { vec![] };
    //println!("{}", json);


     */
    /*
    let mut d = gltf::Gltf {
        ..Default::default()
    };
    //d.document.0.accessors.push(gltf::json::Accessor{
    d.document.0.accessors.push(

        d.document.j
        gltf::json::Accessor {
            buffer_view: Some(0),
            byte_offset: 0,
            count: 0,
            component_type: Valid(gltf::json::accessor::GenericComponentType(
                gltf::json::accessor::ComponentType::F32
            )),
            extensions: None,
            extras: Default::default(),
            type_: Valid(gltf::json::accessor::Type::Vec3),
            min: None,
            max: None,
            name: None,
            normalized: false,
            sparse: None,
        });

     */


    // End test code

    let mut b = BWX::new();
    //let _c = b.load_from_file("EXTERNAL_UI_DEFAULT.PNX")?;
    //let c = b.load_from_file("MON045_DEADA.PNX");
    //let _c = b.load_from_file("MON039_DEFAULTA.PNX")?;
    //let _c = b.load_from_file("MON006_DEFAULTA.PNX");
    let _c = b.load_from_file("MON001_01_DEFAULTA.PNX")?;
    //let _c = b.load_from_file("OBO020_DEFAULT.PNX")?;

    /*

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

     */

    Ok(())
}
