use std::io::Cursor;
use std::mem;
use byteorder::{LittleEndian, WriteBytesExt};
use sl::BWX;
use tracing::{debug, error, Level};
use tracing_subscriber::filter::EnvFilter;
use gltf::json::{self, validation::Checked::Valid};
use serde_json::json;

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
    // let _c = b.load_from_file("MON083_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("HEROSANDRA_DEFAULT.PNX")?;
    // let _c = b.load_from_file("HEROETO_DEFAULT.PNX")?;
    // let _c = b.load_from_file("MON001_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/MONSTER/MON001.PNX/MON001_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/INTERFACE/Login_stage.pnx/LOGIN_STAGE_DEFAULT.PNX")?;

    // Vertex Animation Test Case
    // let _c = b.load_from_file("Assets/Graphic/MONSTER/MON059.PNX/MON059_ATTACKA.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/MONSTER/MON059.PNX/MON059_ATTACKB.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/MONSTER/MON059.PNX/MON059_DEFAULTA.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/MONSTER/MON059.PNX/MON059_IDLEA.PNX")?;
    // let _c = b.load_from_file("../sl2_data/Graphic/MONSTER/MON059.PNX/MON059_ATTACKB.PNX")?;

    // Character: Hero Sandra
    let _c = b.load_from_file("Assets/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_WALK.PNX")?;
    // let _c = b.load_from_file("../sl2_data/Graphic/NPC/WORLD01/HEROSANDRA.PNX/HEROSANDRA_DEFAULT.PNX")?;
    // let _c = b.load_from_file("Assets/Graphic/NPC/WORLD01/HEROETO.PNX/HEROETO_WALK.PNX")?;

    // let _c = b.load_from_file("Assets/Graphic/INTERFACE/Login_infogirl.pnx/LOGIN_INFOGIRL_DEFAULTA.PNX")?;

    save_gltf(&mut b);

    Ok(())
}

fn save_gltf(bwx: &mut BWX) {
    /*
    buffer: Vec<u8>,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    meshes: Vec<json::Mesh>,
    nodes: Vec<json::Node>,
    images: Vec<json::Image>,
    textures: Vec<json::Texture>,
    materials: Vec<json::Material>,
    // Store the material group information
    material_index: Vec<Vec<u32>>,
    node_index: Vec<u32>,
    // animations: Vec<json::Animation>,
    samplers: Vec<json::animation::Sampler>,
    channels: Vec<json::animation::Channel>,

    debug!("{:?}", bwx.head);
    debug!("{:#?}", bwx.objects);
    debug!("{:#?}", bwx.materials);

     */
    let mut buffer: Vec<u8> = vec![];
    let mut buffer_views: Vec<json::buffer::View> = vec![];
    let mut accessors: Vec<json::Accessor> = vec![];
    let mut meshes: Vec<json::Mesh> = vec![];
    let mut nodes: Vec<json::Node> = vec![];
    let mut images: Vec<json::Image> = vec![];
    let mut textures: Vec<json::Texture> = vec![];
    let mut materials: Vec<json::Material> = vec![];
    let mut node_indices: Vec<u32> = vec![];
    let mut samplers: Vec<json::animation::Sampler> = vec![];
    let mut channels: Vec<json::animation::Channel> = vec![];

    for o in &bwx.objects {
        debug!("Name: {:?}, Material: {}", o.name, o.material);
        for m in &o.meshes {
            let mut node_index = vec![];
            debug!("\tMesh material: {}", m.sub_material);

            // Process material
            let material_group = &mut bwx.materials[o.material as usize];
            let sub_material = &mut material_group.sub_materials[m.sub_material as usize];
            if !sub_material.used {
                sub_material.used = true;
                sub_material.material_id = materials.len() as u32;

                // Have texture
                let color_texture = if sub_material.filename.is_some() {
                    let texture_index = textures.len() as u32;
                    let image_index = images.len() as u32;
                    let image = json::Image {
                        buffer_view: None,
                        mime_type: None,
                        name: None,
                        uri: sub_material.filename.clone(),
                        extensions: None,
                        extras: Default::default(),
                    };
                    images.push(image);
                    let texture = json::Texture {
                        name: None,
                        sampler: None,
                        source: json::Index::new(image_index),
                        extensions: None,
                        extras: Default::default(),
                    };
                    textures.push(texture);
                    Some(json::texture::Info {
                        index: json::Index::new(texture_index),
                        tex_coord: 0, // Only have texture_0 now
                        extensions: None,
                        extras: Default::default(),
                    })
                } else { None };

                materials.push(json::Material {
                    alpha_cutoff: None,
                    alpha_mode: Default::default(),
                    double_sided: true,
                    name: Some(material_group.name.clone()),
                    pbr_metallic_roughness: json::material::PbrMetallicRoughness {
                        base_color_factor: Default::default(),
                        base_color_texture: color_texture.clone(),
                        metallic_factor: json::material::StrengthFactor(0.0),
                        roughness_factor: json::material::StrengthFactor(1.0),
                        metallic_roughness_texture: None,
                        extensions: None,
                        extras: Default::default(),
                    },
                    normal_texture: None,
                    occlusion_texture: None,
                    emissive_texture: None,
                    emissive_factor: Default::default(),
                    extensions: None,
                    // "KHR_materials_pbrSpecularGlossiness" extension
                    /*
                    extensions: Some(json::extensions::material::Material {
                        pbr_specular_glossiness: Some(json::extensions::material::PbrSpecularGlossiness {
                            diffuse_factor: Default::default(),
                            diffuse_texture: color_texture,
                            specular_factor: json::extensions::material::PbrSpecularFactor([0.0, 0.0, 0.0]),
                            glossiness_factor: json::material::StrengthFactor(0.0),
                            specular_glossiness_texture: None,
                            extras: Default::default(),
                        })
                    }),
                     */
                    // Not working due to glTF 1.0.0 bug that cannot enable "KHR_materials_specular"
                    /*
                    extensions: Some(json::extensions::material::Material {
                        specular: Some(json::extensions::material::Specular {
                            specular_factor: json::extensions::material::SpecularFactor(0.0),
                            specular_texture: None,
                            specular_color_factor: Default::default(),
                            specular_color_texture: None,
                            extras: (),
                        })
                    }),
                     */
                    extras: Default::default(),
                });
            }

            // Process Index Buffer
            let mut index_buffer = Cursor::new(vec![]);
            for i in &m.indices {
                index_buffer.write_u16::<LittleEndian>(*i);
            }
            // Index buffer might need padding when using u16 (2 bytes)
            let index_buffer_length = index_buffer.get_ref().len();
            let padding = ((index_buffer_length + 3) & !3) - index_buffer_length;
            if padding > 0 {
                debug!("============Padding: {}", padding);
                for _ in 0..padding {
                    index_buffer.write_u8(0);
                }
            }

            // Process Vertex Buffer
            if m.sub_meshes.len() > 1 {
                // FIXME: Only process the first frame of the animation
                error!("Cannot handle vertex animation!, {}@{}", file!(), line!());
            }

            // for sm in &m.sub_meshes
            let sm = &m.sub_meshes[0];
            {
                let mut buffer_view_index = buffer_views.len();
                let mut accessor_index = accessors.len() as u32;
                let mesh_index = meshes.len() as u32;

                // Store the children
                node_index.push(nodes.len() as u32);
                let node = json::Node {
                    camera: None,
                    children: None,
                    extensions: Default::default(),
                    extras: Default::default(),
                    matrix: None,
                    mesh: Some(json::Index::new(mesh_index)),
                    name: None,
                    rotation: None,
                    scale: None,
                    translation: None,
                    skin: None,
                    weights: None,
                };
                nodes.push(node);

                // Mesh - Primitive
                let primitive = json::mesh::Primitive {
                    attributes: {
                        let mut map = std::collections::HashMap::new();
                        map.insert(Valid(json::mesh::Semantic::Positions), json::Index::new(accessor_index));
                        accessor_index += 1;
                        // TODO: Enable normal later
                        // map.insert(Valid(json::mesh::Semantic::Normals), json::Index::new(accessor_index ));
                        // accessor_index += 1;
                        map.insert(Valid(json::mesh::Semantic::TexCoords(0)), json::Index::new(accessor_index));
                        accessor_index += 1;
                        map
                    },
                    extensions: Default::default(),
                    extras: Default::default(),
                    indices: Some(json::Index::new(accessor_index)),
                    material: if o.material < 0 { None } else {
                        Some(json::Index::new(sub_material.material_id))
                    },
                    mode: Valid(json::mesh::Mode::Triangles),
                    targets: None,
                };

                let mesh = json::Mesh {
                    extensions: Default::default(),
                    extras: Default::default(),
                    // name: Some(o.name.clone() + "_Mesh"),
                    // As a mesh group, no name is giving to the mesh but the object
                    name: None,
                    primitives: vec![primitive],
                    weights: None,
                };
                meshes.push(mesh);

                // Vertex
                let mut v_min = [0.0f32, 0.0, 0.0];
                let mut v_max = [0.0f32, 0.0, 0.0];
                let mut v_set = false;
                let mut vertex_buffer = Cursor::new(vec![]);
                debug!("Vertex: {:#?}", sm.vertices.len());
                for v in &sm.vertices {
                    // Write position
                    vertex_buffer.write_f32::<LittleEndian>(v.position[0]);
                    vertex_buffer.write_f32::<LittleEndian>(v.position[1]);
                    vertex_buffer.write_f32::<LittleEndian>(v.position[2]);
                    // TODO: "Write normal" but should not be used before programmed normal calculation
                    // Disable normal output
                    // vertex_buffer.write_f32::<LittleEndian>(v.normal[0]);
                    // vertex_buffer.write_f32::<LittleEndian>(v.normal[1]);
                    // vertex_buffer.write_f32::<LittleEndian>(v.normal[2]);
                    // Write texture coordinate
                    vertex_buffer.write_f32::<LittleEndian>(v.tex_coord[0]);
                    vertex_buffer.write_f32::<LittleEndian>(v.tex_coord[1]);

                    if v_set {
                        let v = v.position;
                        if v[0] > v_max[0] { v_max[0] = v[0]; }
                        if v[1] > v_max[1] { v_max[1] = v[1]; }
                        if v[2] > v_max[2] { v_max[2] = v[2]; }
                        if v[0] < v_min[0] { v_min[0] = v[0]; }
                        if v[1] < v_min[1] { v_min[1] = v[1]; }
                        if v[2] < v_min[2] { v_min[2] = v[2]; }
                    } else {
                        v_min = v.position;
                        v_max = v.position;
                        v_set = true;
                    }
                }
                debug!("Min = {:?}, Max = {:?}", v_min, v_max);

                let mut accessor = json::Accessor {
                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                    byte_offset: 0,
                    count: sm.vertices.len() as u32,
                    component_type: Valid(json::accessor::GenericComponentType(
                        json::accessor::ComponentType::F32
                    )),
                    extensions: None,
                    extras: Default::default(),
                    type_: Valid(json::accessor::Type::Vec3),
                    min: Some(json!(v_min)),
                    max: Some(json!(v_max)),
                    name: None,
                    normalized: false,
                    sparse: None,
                };
                accessors.push(accessor.clone());

                // TODO: Enable normal
                // Normal
                // accessor.byte_offset = (3 * mem::size_of::<f32>()) as u32;
                accessor.min = None;
                accessor.max = None;
                // accessors.push(accessor.clone());

                // Texture Coordinate
                accessor.byte_offset = (3 * mem::size_of::<f32>()) as u32;
                // accessor.byte_offset = (6 * mem::size_of::<f32>()) as u32;
                // Changed value to 3 since there's no normal data
                accessor.type_ = Valid(json::accessor::Type::Vec2);
                accessors.push(accessor.clone());

                let vertex_size = 5 * mem::size_of::<f32>();
                // Changed value to 5 since there's no normal data
                // let vertex_size = 8 * mem::size_of::<f32>();
                let mut buffer_view = json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: vertex_buffer.get_ref().len() as u32,
                    byte_offset: Some(buffer.len() as u32),
                    byte_stride: Some(vertex_size as u32),
                    name: None,
                    target: None,
                    extensions: None,
                    extras: Default::default(),
                };
                buffer_views.push(buffer_view.clone());

                // Index Buffer ------------------
                buffer_view_index = buffer_views.len();
                // Accessor for index
                let accessor = json::Accessor {
                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                    byte_offset: 0,
                    count: m.indices.len() as u32,
                    component_type: Valid(json::accessor::GenericComponentType(
                        json::accessor::ComponentType::U16
                    )),
                    extensions: None,
                    extras: Default::default(),
                    type_: Valid(json::accessor::Type::Scalar),
                    min: None,
                    max: None,
                    name: None,
                    normalized: false,
                    sparse: None,
                };
                accessors.push(accessor);
                // Index bufferView
                // buffer_view_index = self.buffer_views.len();

                debug!("Buffer len: {}", vertex_buffer.get_ref().len());
                buffer.append(vertex_buffer.get_mut());

                buffer_view.buffer = json::Index::new(0);
                buffer_view.byte_length = index_buffer.get_ref().len() as u32;
                buffer_view.byte_offset = Some(buffer.len() as u32);
                buffer_view.byte_stride = None;
                buffer_views.push(buffer_view);

                // Store index buffer to binary
                buffer.append(index_buffer.get_mut());

                let matrix = o.matrices[0].matrix.clone();
                let matrix = gltf::scene::Transform::Matrix {
                    matrix: [
                        [matrix[0], matrix[1], matrix[2], matrix[3]],
                        [matrix[4], matrix[5], matrix[6], matrix[7]],
                        [matrix[8], matrix[9], matrix[10], matrix[11]],
                        [matrix[12], matrix[13], matrix[14], matrix[15]],
                    ]
                };
                let (translation, rotation, scale) = matrix.decomposed();
                // let translation = [t[0], -t[2], t[1]];
                // let rotation = [r[0], -r[2], r[1], r[3]];
                // let scale = [s[0], s[2], s[1]];

                // Store the node for Scene
                let node_count = nodes.len() as u32;
                debug!("mesh_node: {:#?}", node_count);
                debug!("node_index: {:?}", node_index);
                node_indices.push(node_count);
                let node = json::Node {
                    camera: None,
                    children: Some(node_index.into_iter().map(json::Index::new).collect()),
                    extensions: Default::default(),
                    extras: Default::default(),
                    // matrix: Some(node_matrix),
                    matrix: None,
                    mesh: None,
                    name: Some(o.name.clone()),
                    rotation: Some(json::scene::UnitQuaternion(rotation)),
                    scale: Some(scale),
                    translation: Some(translation),
                    skin: None,
                    weights: None,
                };
                nodes.push(node);

                // Matrices
                let mut timeline_max = 0.0;
                let mut o_buffer = Cursor::new(vec![]);
                for mm in &o.matrices {
                    let timeline = mm.timeline as f32 / 3600.0;
                    let m = mm.matrix.clone();
                    let m = gltf::scene::Transform::Matrix {
                        matrix: [
                            [m[0], m[1], m[2], m[3]],
                            [m[4], m[5], m[6], m[7]],
                            [m[8], m[9], m[10], m[11]],
                            [m[12], m[13], m[14], m[15]],
                        ]
                    };
                    // let (translation, rotation, scale) = m.decomposed();
                    // let translation = [t[0], -t[2], t[1]];
                    // let rotation = [r[0], -r[2], r[1], r[3]];
                    // let scale = [s[0], s[2], s[1]];

                    let (translation, rotation, scale) = m.decomposed();
                    // Write timeline, translation, rotation and scale to buffer
                    // Could use system's array.as_bytes, but cannot ensure when running on big endian system
                    // So use the old school byteorder method
                    o_buffer.write_f32::<LittleEndian>(timeline);
                    for v in translation { o_buffer.write_f32::<LittleEndian>(v); }
                    for v in rotation { o_buffer.write_f32::<LittleEndian>(v); }
                    for v in scale { o_buffer.write_f32::<LittleEndian>(v); }

                    if timeline > timeline_max {
                        timeline_max = timeline;
                    }
                }
                // Store data in buffer
                let offset = buffer.len();
                let length = o_buffer.get_ref().len();
                buffer.append(o_buffer.get_mut());

                // Prepare bufferView
                let buffer_view_index = buffer_views.len();
                let buffer_view = json::buffer::View {
                    buffer: json::Index::new(0),
                    byte_length: length as u32,
                    byte_offset: Some(offset as u32),
                    byte_stride: Some((mem::size_of::<f32>() * 11) as u32),
                    name: Some(o.name.clone() + "_Matrix"),
                    target: None,
                    extensions: None,
                    extras: Default::default(),
                };
                buffer_views.push(buffer_view);

                // let animation_count = matrix.len() as u32 - 1;
                let animation_count = o.matrices.len() as u32;
                // Accessor for timeline
                let accessor_index = accessors.len();
                let accessor = json::Accessor {
                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                    byte_offset: 0,
                    count: animation_count,
                    component_type: Valid(json::accessor::GenericComponentType(
                        json::accessor::ComponentType::F32)),
                    extensions: None,
                    extras: Default::default(),
                    type_: Valid(json::accessor::Type::Scalar),
                    min: Some(json!([0.0f32])),
                    max: Some(json!([timeline_max])),
                    name: Some(o.name.clone() + "_Timeline"),
                    normalized: false,
                    sparse: None,
                };
                accessors.push(accessor);
                debug!("bufferView: {}", buffer_view_index);

                // Accessor for Translation
                let mut accessor = json::Accessor {
                    buffer_view: Some(json::Index::new(buffer_view_index as u32)),
                    byte_offset: mem::size_of::<f32>() as u32,
                    count: animation_count,
                    component_type: Valid(json::accessor::GenericComponentType(
                        json::accessor::ComponentType::F32)),
                    extensions: None,
                    extras: Default::default(),
                    type_: Valid(json::accessor::Type::Vec3),
                    min: None,
                    max: None,
                    name: Some(o.name.clone() + "_Translation"),
                    normalized: false,
                    sparse: None,
                };
                accessors.push(accessor.clone());
                // Accessor for Rotation
                accessor.byte_offset = (1 + 3) * mem::size_of::<f32>() as u32;
                accessor.type_ = Valid(json::accessor::Type::Vec4);
                accessor.name = Some(o.name.clone() + "_Rotation");
                accessors.push(accessor.clone());
                // Accessor for Scale
                accessor.byte_offset = (1 + 3 + 4) * mem::size_of::<f32>() as u32;
                accessor.type_ = Valid(json::accessor::Type::Vec3);
                accessor.name = Some(o.name.clone() + "_Scale");
                accessors.push(accessor);


                // Samplers
                // let mut samplers = vec![];
                let sampler_index = samplers.len() as u32;
                // Samplers - Translation
                let mut sampler = json::animation::Sampler {
                    extensions: None,
                    extras: Default::default(),
                    input: json::Index::new(accessor_index as u32),
                    interpolation: Valid(json::animation::Interpolation::Linear),
                    output: json::Index::new(accessor_index as u32 + 1),
                };
                samplers.push(sampler.clone());
                // Samplers - Rotation
                sampler.output = json::Index::new(accessor_index as u32 + 2);
                samplers.push(sampler.clone());
                // Samplers - Scale
                sampler.output = json::Index::new(accessor_index as u32 + 3);
                samplers.push(sampler);

                // Channels
                // let mut channels = vec![];
                // Channel - Translation
                let mut channel = json::animation::Channel {
                    sampler: json::Index::new(sampler_index),
                    target: json::animation::Target {
                        extensions: None,
                        extras: Default::default(),
                        node: json::Index::new(node_count),
                        path: Valid(json::animation::Property::Translation),
                    },
                    extensions: None,
                    extras: Default::default(),
                };
                channels.push(channel.clone());
                // Channel - Rotation
                channel.sampler = json::Index::new(sampler_index + 1);
                channel.target.path = Valid(json::animation::Property::Rotation);
                channels.push(channel.clone());
                // Channel - Scale
                channel.sampler = json::Index::new(sampler_index + 2);
                channel.target.path = Valid(json::animation::Property::Scale);
                channels.push(channel);
            }
        }
    }

    //debug!("{:#?}", self.data);

    let oname = "test".to_string();

    let buffer_json = json::Buffer {
        byte_length: buffer.len() as u32,
        extensions: Default::default(),
        extras: Default::default(),
        name: None,
        uri: Some(oname.clone() + ".bin"),
    };

    let asset = json::Asset {
        copyright: Some("SLODT All Rights Reserved. (C) 2003-2022".into()),
        extensions: None,
        extras: Default::default(),
        generator: Some("Tommy's BWX Exporter".into()),
        min_version: None,
        version: "2.0".to_string(),
    };

    let scene_nodes = node_indices.iter()
        .map(|x| json::Index::new(*x)).collect();

    // Disable sampler should display correct texture
    /*
    let sampler = json::texture::Sampler {
        mag_filter: Some(Valid(MagFilter::Nearest)),
        min_filter: Some(Valid(MinFilter::Nearest)),
        name: None,
        wrap_s: Valid(WrappingMode::ClampToEdge),
        wrap_t: Valid(WrappingMode::ClampToEdge),
        extensions: None,
        extras: Default::default(),
    };
     */

    let root_node_index = nodes.len() as u32;
    let node = json::Node {
        camera: None,
        children: Some(scene_nodes),
        extensions: Default::default(),
        extras: Default::default(),
        matrix: None,
        mesh: None,
        name: Some("Root".into()),
        // Rotate -90 degrees along X axis
        rotation: Some(json::scene::UnitQuaternion([-0.707, 0.0, 0.0, 0.707])),
        // Scale to 0.1
        scale: Some([0.1, 0.1, 0.1].into()),
        translation: None,
        skin: None,
        weights: None,
    };
    nodes.push(node);

    let root = json::Root {
        asset,
        scene: Some(json::Index::new(0)),
        scenes: vec![json::Scene {
            extensions: Default::default(),
            extras: Default::default(),
            name: Some("Scene".into()),
            // nodes: scene_nodes,
            nodes: vec![json::Index::new(root_node_index)],
        }],
        nodes,
        meshes,
        accessors,
        buffer_views,
        buffers: vec![buffer_json],
        //samplers: vec![sampler],
        materials,
        textures,
        images,
        animations: vec![json::Animation {
            extensions: None,
            extras: Default::default(),
            channels,
            name: Some("Animation".into()),
            samplers,
        }],
        ..Default::default()
    };

    let j = json::serialize::to_string_pretty(&root).expect("OK");
    //debug!("glTF:\n{}", j);

    std::fs::write(oname.clone() + ".gltf", j.as_bytes());
    std::fs::write(oname + ".bin", buffer.clone());

    // debug!("{:#?}", self.material_index);


    /*
    for m in &bwx.materials {
        debug!("Material: {}", m.name);
        for sm in &m.sub_materials {
            debug!("\tSub_Material: {:?}", sm.filename);
        }
    }
     */

    // debug!("{:#?}", bwx.materials[0].sub_materials[1]);
    // debug!("{:#?}", bwx.materials[9].sub_materials[0]);
}
