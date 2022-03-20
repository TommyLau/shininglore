use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};
use tracing::{debug, error, info, trace, warn};
use serde::Serialize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
#[derive(Debug, Clone, Serialize)]
pub enum SlType {
    UChar(u8),
    Char(i8),
    Word(i16),
    Int(i32),
    Float(f32),
    String(String),
    Data(Vec<u8>),
    Array(Vec<SlType>),
    DArray(String, Vec<SlType>),
    None,
}

impl Default for SlType {
    fn default() -> Self {
        SlType::None
    }
}

impl SlType {
    pub fn int(&self) -> Result<i32> {
        match self {
            SlType::UChar(v) => Ok(*v as i32),
            SlType::Char(v) => Ok(*v as i32),
            SlType::Word(v) => Ok(*v as i32),
            SlType::Int(v) => Ok(*v),
            _ => Err("Cannot get integer from SlType".into()),
        }
    }

    pub fn float(&self) -> Result<f32> {
        if let SlType::Float(v) = self {
            Ok(*v)
        } else {
            Err("Cannot get float from SlType".into())
        }
    }

    pub fn string(&self) -> Result<String>
    {
        match self {
            SlType::String(v) | SlType::DArray(v, _) => { Ok(v.into()) }
            _ => Err("Cannot get string from SlType".into())
        }
    }

    pub fn data(&self) -> Result<&Vec<u8>> {
        if let SlType::Data(v) = self {
            Ok(v)
        } else {
            Err("Cannot get data from SlType".into())
        }
    }

    pub fn array(&self) -> Result<&Vec<SlType>> {
        if let SlType::Array(v) = self {
            Ok(v)
        } else {
            Err("Cannot get array from SlType".into())
        }
    }

    pub fn d_array(&self) -> Result<&Vec<SlType>> {
        if let SlType::DArray(_, v) = self {
            Ok(v)
        } else {
            Err("Cannot get d-array from SlType".into())
        }
    }
}

// Block 'HEAD'
pub struct Head {
    pub name: String,
    pub description: String,
    pub magic: i32,
    pub version: i16,
    pub other: String,
}

pub struct SubMaterial {
    pub highlight: f32,
    pub filename: String,
    pub used: bool,
}

// Block 'MTRL'
pub struct Material {
    pub sub_materials: Vec<SubMaterial>,
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    pub data: SlType,
    version: i32,
}

impl BWX {
    /// Returns a BWX with the given file name
    ///
    // /// TODO: Merge load_from_file() to new()
    // /// # Arguments
    // ///
    // /// * `path` - A string slice that holds the file name of a BNX / PNX file
    // ///
    /// # Examples
    ///
    /// ```
    /// use bwx::BWX;
    /// let bwx = BWX::new();
    /// ```
    pub fn new() -> Self {
        BWX {
            ..Default::default()
        }
    }

    /// Load BWX file from file
    #[tracing::instrument(skip(self, filename))]
    pub fn load_from_file(&mut self, filename: &str) -> Result<()> {
        info!(filename);

        let data = std::fs::read(filename)?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);
        self.data = self.go_through(true)?;

        for node in self.data.array().unwrap() {
            let name = node.string()?;
            let children = node.d_array()?;
            match name.as_str() {
                "0" => {
                    // Default block "0" with string "SLBWX"
                }
                "HEAD" => {
                    if children.len() >= 4 {
                        // 0 - HEAD
                        // 1 - head block
                        // 2 - PNX\0
                        // 3 - 0x0500: SL1, 0x0602: SL2
                        // 4 - "BWX PNX KAK"
                        if children[0].string().unwrap().as_str() != "HEAD" {
                            error!("Incorrect HEAD block");
                        }
                        if children[2].int().unwrap() != 0x504e5800 {
                            error!("Header magic != PNX");
                        }
                        self.version = children[3].int()?;
                        match self.version {
                            0x500 => trace!("ShiningLore V1 PNX"),
                            0x602 => trace!("ShiningLore V2 PNX"),
                            _ => error!("Unknown ShiningLore PNX version!!"),
                        }
                    } else {
                        warn!("HEAD block length < 4, no PNX version available!");
                    }
                }
                "MTRL" => {
                    // TODO: Save material information in struct
                    for materials in children {
                        let material = materials.array()?;
                        // material[0] - Material Group "MTRL"
                        // material[1] - Material Group Name
                        // material[2..n] - Material Array for Sub Materials
                        trace!("Material: {}", material[1].string()?);
                        for (i, sub_materials) in material.iter().enumerate().skip(2) {
                            let sub_material = sub_materials.array()?;
                            let highlight = sub_material[5].float()?;
                            // 0 - "SUBMTRL"
                            // 1 - Diffuse ???
                            // 2 - Ambient ???
                            // 3 - Specular ???
                            // 4 - Some float ???
                            // 5 - High light
                            // 6 - Most 0x01
                            // 7 - ???
                            // 8 - Texture Array
                            let filename = if sub_material.len() > 8 {
                                // Some materials have no texture, such as glass
                                let texture = sub_material[8].array()?;
                                // 0 - TEX
                                // 1 - Most 0x00, timer?
                                // 2 - Filename
                                let filename = texture[2].string()?;
                                filename.split('\\').last().unwrap().into()
                            } else { "".to_string() };
                            trace!("\tSub Material {}: Highlight: {}, File: {}", i - 2, highlight, filename);
                        }
                    }
                }
                // TODO: Parse OBJ2 mesh data from SL1
                "OBJ2" => {
                    warn!("OBJ2 parsing needs to be implemented! {}@{}", file!(), line!());
                }
                "OBJECT" => {}
                "CAM" => {}
                "LIGHT" => {}
                "SOUND" => {}
                "BONE" => {}
                "CHART" => {}
                "DXOBJ" | "SPOB" => {
                    // TODO: Store parsed data into struct
                    if children.is_empty() {
                        warn!("No data block found in {}", name);
                        continue;
                    }

                    for objects in children {
                        let object = objects.array()?;
                        // 0 - "DXOBJ" / "SPOB"
                        // 1 - Mesh Name
                        // 2 - Unknown integer
                        // 3 - Texture Group Index : -1 means no texture
                        // 4, 5 - Unknown
                        // 6 - 0x4D534858h("MSHX") or 0x4D4E4858h("MNHX")
                        // 7 - Array("DXMESH")
                        // 8 - Array("MATRIX")
                        let name = object[1].string()?;
                        let texture_index = object[3].int()?;
                        trace!("Object: {}, Index: {}", name, texture_index);
                        // Meshes
                        let meshes = object[7].array()?;
                        for mesh_array in meshes {
                            let mesh = mesh_array.array()?;
                            // 0 - "DXMESH"
                            // 1 - Texture Index in Texture Group
                            // 2 - Array("DXMESHF")
                            // 3 - Index Buffer Size
                            // 4 - Index Buffer
                            let sub_texture_index = mesh[1].int()?;
                            let index_count = mesh[3].int()?;
                            let index_buffer = mesh[4].data()?;
                            if index_buffer.len() != index_count as usize * 2 {
                                error!("Index block size incorrect!");
                            }
                            trace!("\tMesh: [Texture Index: {}, Index Count: {}", sub_texture_index, index_count);
                            let blocks = mesh[2].array()?;
                            for vertices in blocks {
                                let vertex = vertices.array()?;
                                // 0 - "DXMESHF"
                                // 1 - VB Timer
                                // 2 - Vertex Type??? - 0x15
                                // 3 - Vertex Count
                                // 4 - Vertex Size - 0x20
                                // 5 - Vertex Buffer
                                let timer = vertex[1].int()?;
                                let vertex_type = vertex[2].int()?;
                                let vertex_count = vertex[3].int()?;
                                let vertex_size = vertex[4].int()?;
                                let vertex_buffer = vertex[5].data()?;
                                trace!("\t\tVertex: [Timer: {}, Type: {}, Count: {}, Size: {}, BufLen: {}",
                                timer, vertex_type, vertex_count, vertex_size, vertex_buffer.len());
                            }
                        }
                        // Matrices
                        let matrices = object[8].array()?;
                        for matrix_array in matrices {
                            let matrix = matrix_array.array()?;
                            // 0 - "MATRIX"
                            // 1..n - Matrix
                            // TODO: Parse matrix later
                            trace!("\tMatrix: [Length: {}]", matrix.len() -1);
                        }
                        // SFX Blocks?
                        if object.len() > 9 {
                            // TODO: Parse SFX
                            let sfx = object[9].array()?;
                            if !sfx.is_empty() {
                                warn!("\tSFX: Unhandled SFX blocks? {}@{}", file!(), line!());
                            }
                        }
                    }
                }
                _ => {
                    error!("Unknown block: {}", name);
                }
            }
        }

        Ok(())
    }

    /// Check whether the file is a valid BNX / PNX format
    fn check_bwx_header(&mut self) -> Result<()> {
        let header = &self.content.get_ref()[..4];
        if header != "BWXF".as_bytes() {
            return Err("Invalid BWX file.".into());
        }

        Ok(())
    }

    /// Read BNX / PNX special packed integer value (little endian)
    fn read_i32_packed(&mut self) -> Result<i32> {
        let mut result: u32 = 0;
        let mut shift = 0;

        while shift < 35 {
            let t = self.content.read_u8()? as u32;
            result |= (t & 0x7f) << shift;

            if t & 0x80 == 0 {
                break;
            } else {
                shift += 7;
            }
        }

        Ok(result as i32)
    }

    /// Read block size & numbers
    fn read_block_size_number(&mut self) -> Result<(i32, i32)> {
        Ok((self.read_i32_packed()?, self.read_i32_packed()?))
    }

    /// Read string
    fn read_string(&mut self) -> Result<String> {
        let length = self.content.read_u8()?;
        let mut buffer = Vec::new();
        buffer.resize(length as usize, 0);
        self.content.read_exact(&mut buffer)?;

        let (cow, _encoding, had_errors) = encoding_rs::EUC_KR.decode(&buffer);
        if had_errors {
            error!("Failed to convert string from Korean to UTF-8!");
            Ok(String::from_utf8_lossy(&buffer).trim_matches('\0').to_string())
        } else {
            Ok(cow.trim_matches('\0').to_string())
        }
    }

    /// Go through the whole BWX file
    #[tracing::instrument(skip(self, root))]
    fn go_through(&mut self, root: bool) -> Result<SlType> {
        // Parse root as Signature D
        let signature = if root { 0x44 } else { self.content.read_u8()? };

        let data = match signature {
            0x41 => { // Signature A
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature A] - Size: {}, Blocks: {}", size, blocks);
                let mut node = vec![];
                while blocks > 0 {
                    let child = self.go_through(false)?;
                    node.push(child);
                    blocks -= 1;
                }
                SlType::Array(node)
            }
            0x42 => { // Signature B
                let size = self.read_i32_packed()?;
                trace!("[Signature B] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size as usize, 0);
                self.content.read_exact(&mut buffer)?;
                SlType::Data(buffer)
            }
            0x43 => { // Signature C
                let value = -self.content.read_i8()?;
                trace!("[Signature C] - Value: {}", value);
                SlType::Char(value)
            }
            0x44 => { // Signature D
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature D] - Size: {}, Blocks: {}", size, blocks);
                let mut node = vec![];
                while blocks > 0 {
                    let name = self.read_string()?;
                    trace!("[Signature D] - Name: {}", name);
                    let child = self.go_through(false)?;
                    match child {
                        SlType::Array(children) => {
                            node.push(SlType::DArray(name, children));
                        }
                        _ => {
                            node.push(SlType::DArray(name, vec![child]));
                        }
                    }
                    blocks -= 1;
                }
                SlType::Array(node)
            }
            0x46 => { // Signature F
                let value = self.content.read_f32::<LittleEndian>()?;
                trace!("[Signature F] - Value: {:.3}", value);
                SlType::Float(value)
            }
            0x48 => { // Signature H
                let value = -self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature H] - Value: {}", value);
                SlType::Word(value)
            }
            0x49 => { // Signature I
                let value = self.content.read_i32::<LittleEndian>()?;
                trace!("[Signature I] - Value: {}", value);
                SlType::Int(value)
            }
            0x53 => { // Signature S
                let value = self.read_string()?;
                trace!("[Signature S] - Value: {}", value);
                SlType::String(value)
            }
            0x57 => { // Signature W
                let value = self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature W] - Value: {}", value);
                SlType::Word(value)
            }
            0x59 => { // Signature Y
                let value = self.content.read_u8()?;
                trace!("[Signature Y] - Value: {}", value);
                SlType::UChar(value)
            }
            s if s < 0x20 => {
                // Independent data
                trace!("[Independent Data] - Value: {}", s);
                SlType::UChar(s)
            }
            s  if s >= 0x80 => {
                // Independent data block
                let size = s as usize & 0x7f;
                trace!("[Independent Data Block] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size, 0);
                self.content.read_exact(&mut buffer)?;
                SlType::Data(buffer)
            }
            _ => {
                error!("Unhandled signature = 0x{:02x}, position: {}", signature, self.content.position());
                //debug!("{:#?}", self.node);
                panic!("Unhandled type {}", signature);
            }
        };
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_bwx_header() {
        let mut bwx = BWX::new();
        bwx.content = Cursor::new(vec![66, 87, 88, 70]);
        assert!(bwx.check_bwx_header().is_ok(), "File header check should pass");
        bwx.content = Cursor::new(vec![11, 22, 33, 44]);
        assert!(bwx.check_bwx_header().is_err(), "File header check should fail");
    }

    #[test]
    fn read_i32_packed() {
        let mut bwx = BWX::new();
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0xc1, 0xef, 0x5a, 0x0c]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 1488833, "Packed integer value incorrect");
        bwx.content = Cursor::new(vec![0x0c, 0x02]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 12, "Packed integer value incorrect");
    }

    #[test]
    fn read_string() {
        let mut bwx = BWX::new();
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.content = Cursor::new(vec![0x02, 0x30, 0x00, 0x53]);
        assert_eq!(bwx.read_string().unwrap().as_str(), "0", "The string should be '0'");
    }
}

