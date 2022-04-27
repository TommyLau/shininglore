use crate::bwx::*;
use std::io::Cursor;
use std::mem;
use std::path::PathBuf;
use byteorder::{LittleEndian, WriteBytesExt};
use tracing::{debug, error, Level};
use gltf::json::{self, validation::Checked::Valid};
use serde_json::json;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug, Default)]
pub struct Gltf {
    // Filename
    filename: PathBuf,
    // BWX handler
    bwx: BWX,
    // Final bin file
    buffer: Vec<u8>,
    buffer_views: Vec<json::buffer::View>,
    accessors: Vec<json::Accessor>,
    meshes: Vec<json::Mesh>,
    nodes: Vec<json::Node>,
    images: Vec<json::Image>,
    textures: Vec<json::Texture>,
    materials: Vec<json::Material>,
    node_indices: Vec<u32>,
    samplers: Vec<json::animation::Sampler>,
    channels: Vec<json::animation::Channel>,
}

impl Gltf {
    pub fn new() -> Self {
        Gltf {
            ..Default::default()
        }
    }

    pub fn load_from_bwx<T>(&mut self, filename: T) -> Result<()>
        where T: AsRef<Path> {
        self.filename = filename.as_ref().into();
        /*
        debug!("{:#?}", self.filename);
        self.filename.set_extension("txt");
        debug!("{:#?}", self.filename);
        self.filename.set_file_name("new_file.gltf");
        debug!("{:#?}", self.filename);
        self.filename.pop();
        debug!("{:#?}", self.filename);
        self.filename.push("test.bin");
        debug!("{:#?}", self.filename);
         */

        self.bwx.load_from_file(filename)?;

        Ok(())
    }

    pub fn save_gltf(&mut self) {
        /*
    // Store the material group information
    material_index: Vec<Vec<u32>>,
    // animations: Vec<json::Animation>,

    debug!("{:?}", bwx.head);
    debug!("{:#?}", bwx.objects);
    debug!("{:#?}", bwx.materials);

     */

        for o in &self.bwx.objects {
            debug!("Name: {:?}, Material: {}", o.name, o.material);
            for m in &o.meshes {
                let mut node_index = vec![];
                debug!("\tMesh material: {}", m.sub_material);

                // Process material
                let material_group = &mut self.bwx.materials[o.material as usize];
                let sub_material = &mut material_group.sub_materials[m.sub_material as usize];
                if !sub_material.used {
                    // Material hasn't been generated yet, prepare material and save the material id
                    sub_material.used = true;
                    sub_material.material_id = self.materials.len() as u32;

                    let base_color_texture = if sub_material.filename.is_some() {
                        // Have texture
                        let texture_index = self.textures.len() as u32;
                        let image_index = self.images.len() as u32;
                        let image = prepare_json_image(sub_material.filename.clone());
                        self.images.push(image);
                        let texture = prepare_json_texture(image_index);
                        self.textures.push(texture);
                        Some(prepare_json_texture_info(texture_index))
                    } else { None };

                    let material = prepare_json_material(&material_group.name, base_color_texture);
                    self.materials.push(material);
                }

                // Process Index Buffer
                let mut index_buffer = Cursor::new(vec![]);
                for i in &m.indices {
                    index_buffer.write_u16::<LittleEndian>(*i);
                }
                // Index buffer might need padding when using u16 (2 bytes)
                buffer_padding(&mut index_buffer);

                // Process Vertex Buffer
                if m.sub_meshes.len() > 1 {
                    // FIXME: Only process the first frame of the animation
                    error!("Cannot handle vertex animation!, {}@{}", file!(), line!());
                }

                // for sm in &m.sub_meshes
                let sm = &m.sub_meshes[0];
                {
                    let mut buffer_view_index = self.buffer_views.len();
                    let mut accessor_index = self.accessors.len() as u32;
                    let mesh_index = self.meshes.len() as u32;

                    // Store the children
                    node_index.push(self.nodes.len() as u32);
                    let node = prepare_json_node(mesh_index);
                    self.nodes.push(node);

                    // Mesh - Primitive
                    let primitive = prepare_json_mesh_primitive(
                        accessor_index,
                        accessor_index + 2,
                        if o.material < 0 { None } else {
                            Some(sub_material.material_id)
                        });
                    let mesh = prepare_json_mesh(primitive);
                    self.meshes.push(mesh);

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
                    self.accessors.push(accessor.clone());

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
                    self.accessors.push(accessor.clone());

                    let vertex_size = 5 * mem::size_of::<f32>();
                    // Changed value to 5 since there's no normal data
                    // let vertex_size = 8 * mem::size_of::<f32>();
                    let mut buffer_view = json::buffer::View {
                        buffer: json::Index::new(0),
                        byte_length: vertex_buffer.get_ref().len() as u32,
                        byte_offset: Some(self.buffer.len() as u32),
                        byte_stride: Some(vertex_size as u32),
                        name: None,
                        target: None,
                        extensions: None,
                        extras: Default::default(),
                    };
                    self.buffer_views.push(buffer_view.clone());

                    // Index Buffer ------------------
                    buffer_view_index = self.buffer_views.len();
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
                    self.accessors.push(accessor);
                    // Index bufferView
                    // buffer_view_index = self.buffer_views.len();

                    debug!("Buffer len: {}", vertex_buffer.get_ref().len());
                    self.buffer.append(vertex_buffer.get_mut());

                    buffer_view.buffer = json::Index::new(0);
                    buffer_view.byte_length = index_buffer.get_ref().len() as u32;
                    buffer_view.byte_offset = Some(self.buffer.len() as u32);
                    buffer_view.byte_stride = None;
                    self.buffer_views.push(buffer_view);

                    // Store index buffer to binary
                    self.buffer.append(index_buffer.get_mut());

                    let (translation, rotation, scale) = matrix_decomposed(&o.matrices[0].matrix);

                    // Store the node for Scene
                    let node_count = self.nodes.len() as u32;
                    debug!("mesh_node: {:#?}", node_count);
                    debug!("node_index: {:?}", node_index);
                    self.node_indices.push(node_count);
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
                    self.nodes.push(node);

                    // Matrices
                    let mut timeline_max = 0.0;
                    let mut o_buffer = Cursor::new(vec![]);
                    for mm in &o.matrices {
                        let timeline = mm.timeline as f32 / 3600.0;
                        let (translation, rotation, scale) = matrix_decomposed(&mm.matrix);
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
                    let offset = self.buffer.len();
                    let length = o_buffer.get_ref().len();
                    self.buffer.append(o_buffer.get_mut());

                    // Prepare bufferView
                    let buffer_view_index = self.buffer_views.len();
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
                    self.buffer_views.push(buffer_view);

                    // let animation_count = matrix.len() as u32 - 1;
                    let animation_count = o.matrices.len() as u32;
                    // Accessor for timeline
                    let accessor_index = self.accessors.len();
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
                    self.accessors.push(accessor);
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
                    self.accessors.push(accessor.clone());
                    // Accessor for Rotation
                    accessor.byte_offset = (1 + 3) * mem::size_of::<f32>() as u32;
                    accessor.type_ = Valid(json::accessor::Type::Vec4);
                    accessor.name = Some(o.name.clone() + "_Rotation");
                    self.accessors.push(accessor.clone());
                    // Accessor for Scale
                    accessor.byte_offset = (1 + 3 + 4) * mem::size_of::<f32>() as u32;
                    accessor.type_ = Valid(json::accessor::Type::Vec3);
                    accessor.name = Some(o.name.clone() + "_Scale");
                    self.accessors.push(accessor);


                    // Samplers
                    // let mut samplers = vec![];
                    let sampler_index = self.samplers.len() as u32;
                    // Samplers - Translation
                    let mut sampler = json::animation::Sampler {
                        extensions: None,
                        extras: Default::default(),
                        input: json::Index::new(accessor_index as u32),
                        interpolation: Valid(json::animation::Interpolation::Linear),
                        output: json::Index::new(accessor_index as u32 + 1),
                    };
                    self.samplers.push(sampler.clone());
                    // Samplers - Rotation
                    sampler.output = json::Index::new(accessor_index as u32 + 2);
                    self.samplers.push(sampler.clone());
                    // Samplers - Scale
                    sampler.output = json::Index::new(accessor_index as u32 + 3);
                    self.samplers.push(sampler);

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
                    self.channels.push(channel.clone());
                    // Channel - Rotation
                    channel.sampler = json::Index::new(sampler_index + 1);
                    channel.target.path = Valid(json::animation::Property::Rotation);
                    self.channels.push(channel.clone());
                    // Channel - Scale
                    channel.sampler = json::Index::new(sampler_index + 2);
                    channel.target.path = Valid(json::animation::Property::Scale);
                    self.channels.push(channel);
                }
            }
        }

        let oname = "test".to_string();

        let buffer_json = json::Buffer {
            byte_length: self.buffer.len() as u32,
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

        let scene_nodes = self.node_indices.iter()
            .map(|x| json::Index::new(*x)).collect();

        let root_node_index = self.nodes.len() as u32;
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
        self.nodes.push(node);

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
            nodes: self.nodes.clone(),
            meshes: self.meshes.clone(),
            accessors: self.accessors.clone(),
            buffer_views: self.buffer_views.clone(),
            buffers: vec![buffer_json],
            //samplers: vec![sampler],
            materials: self.materials.clone(),
            textures: self.textures.clone(),
            images: self.images.clone(),
            animations: vec![json::Animation {
                extensions: None,
                extras: Default::default(),
                channels: self.channels.clone(),
                name: Some("Animation".into()),
                samplers: self.samplers.clone(),
            }],
            ..Default::default()
        };

        let j = json::serialize::to_string_pretty(&root).expect("OK");

        std::fs::write("./tmp/".to_owned() + &oname + ".gltf", j.as_bytes());
        std::fs::write("./tmp/".to_owned() + &oname + ".bin", self.buffer.clone());
    }

    fn buffer_append(&mut self) -> Result<()>
    {
        Ok(())
    }
}

/// Decompose matrix
fn matrix_decomposed(matrix: &[f32; 16]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let matrix = gltf::scene::Transform::Matrix {
        matrix: [
            [matrix[0], matrix[1], matrix[2], matrix[3]],
            [matrix[4], matrix[5], matrix[6], matrix[7]],
            [matrix[8], matrix[9], matrix[10], matrix[11]],
            [matrix[12], matrix[13], matrix[14], matrix[15]],
        ]
    };
    matrix.decomposed()
}

/// Buffer padding to 4 bytes alignment
fn buffer_padding(buffer: &mut Cursor<Vec<u8>>) {
    let buffer_length = buffer.get_ref().len();
    let padding = ((buffer_length + 3) & !3) - buffer_length;
    if padding > 0 {
        for _ in 0..padding {
            buffer.write_u8(0);
        }
    }
}

fn prepare_json_image(filename: Option<String>) -> json::Image {
    json::Image {
        buffer_view: None,
        mime_type: None,
        name: None,
        uri: filename,
        extensions: None,
        extras: Default::default(),
    }
}

fn prepare_json_texture(image_index: u32) -> json::Texture {
    json::Texture {
        name: None,
        sampler: None,
        source: json::Index::new(image_index),
        extensions: None,
        extras: Default::default(),
    }
}

fn prepare_json_texture_info(texture_index: u32) -> json::texture::Info {
    json::texture::Info {
        index: json::Index::new(texture_index),
        tex_coord: 0, // Only have texture_0 now
        extensions: None,
        extras: Default::default(),
    }
}

fn prepare_json_material(material_name: &str, base_color_texture: Option<json::texture::Info>) -> json::Material {
    json::Material {
        alpha_cutoff: None,
        alpha_mode: Default::default(),
        // Because the normal faces' error, use double sided texture
        double_sided: true,
        name: Some(material_name.into()),
        pbr_metallic_roughness: json::material::PbrMetallicRoughness {
            base_color_factor: Default::default(),
            base_color_texture,
            // Set metallic to 0, roughness to 1 to display SL characters correctly
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
        extras: Default::default(),
    }
}

fn prepare_json_node(mesh_index: u32) -> json::Node {
    json::Node {
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
    }
}

fn prepare_json_mesh_primitive(accessor_index: u32, indices_index: u32, material_id: Option<u32>) -> json::mesh::Primitive {
    json::mesh::Primitive {
        attributes: {
            let mut map = std::collections::HashMap::new();
            map.insert(Valid(json::mesh::Semantic::Positions), json::Index::new(accessor_index));
            // TODO: Enable normal later
            // map.insert(Valid(json::mesh::Semantic::Normals), json::Index::new(accessor_index + 1));
            map.insert(Valid(json::mesh::Semantic::TexCoords(0)), json::Index::new(accessor_index + 1));
            map
        },
        extensions: Default::default(),
        extras: Default::default(),
        indices: Some(json::Index::new(indices_index)),
        material: if let Some(m) = material_id { Some(json::Index::new(m)) } else { None },
        mode: Valid(json::mesh::Mode::Triangles),
        targets: None,
    }
}

fn prepare_json_mesh(primitive: json::mesh::Primitive) -> json::Mesh {
    json::Mesh {
        extensions: Default::default(),
        extras: Default::default(),
        // As a mesh group, no name is giving to the mesh but the object
        name: None,
        primitives: vec![primitive],
        weights: None,
    }
}
