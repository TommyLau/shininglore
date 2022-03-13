use std::cell::RefCell;
use std::io::{Cursor, Read};
use std::rc::Rc;
use byteorder::{LittleEndian, ReadBytesExt};
use tracing::{debug, error, info, trace, warn};

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
#[derive(Debug)]
pub enum SlData {
    UChar(u8),
    Char(i8),
    Word(i16),
    Int(i32),
    Float(f32),
    String(String),
    BlockSizeNumber(i32, i32),
    Data(Vec<u8>),
    None,
}

#[derive(Debug)]
pub struct Node {
    pub data: SlData,
    pub children: Vec<Rc<RefCell<Node>>>,
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    pub node: Node,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

static mut COUNT: u32 = 0;

impl Default for Node {
    fn default() -> Self {
        Node {
            data: SlData::None,
            children: vec![],
        }
    }
}

impl Node {
    /// Create a new Node, with SlData
    ///
    /// # Arguments
    ///
    /// * data - SlData
    ///
    /// # Examples
    ///
    /// ```
    /// use bwx::{Node, SlData};
    /// let node = Node::new(SlData::None);
    ///```
    pub fn new(data: SlData) -> Self {
        Node {
            data,
            ..Default::default()
        }
    }
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
        BWX { ..Default::default() }
    }

    #[tracing::instrument(skip(self, filename))]
    pub fn load_from_file(&mut self, filename: &str) -> Result<()> {
        trace!(filename);

        let data = std::fs::read(filename)?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);

        let (size, mut blocks) = self.read_block_size_number()?;
        trace!(size, blocks);

        while blocks > 0 {
            let name = self.read_string()?;
            trace!("section name: {}", name);
            let node = Rc::new(RefCell::new(Node {
                children: vec![],
                data: SlData::String(name),
            }));
            let res_node = self.parse_block()?;
            node.borrow_mut().children.push(res_node);
            // Try to parse into list
            self.node.children.push(node);

            blocks -= 1;
        }

        debug!("{:#?}", self.node);

        Ok(())
    }

    /// Check whether the file is a valid BNX / PNX format
    fn check_bwx_header(&mut self) -> Result<()> {
        let header = &self.content.get_ref()[..4];
        if header != "BWXF".as_bytes() {
            return Err("Invalid BWX file.")?;
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

    /// Parse block
    #[tracing::instrument(skip(self))]
    fn parse_block(&mut self) -> Result<Rc<RefCell<Node>>> {
        let signature = self.content.read_u8()?;

        match signature {
            0x41 => { // Signature A
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature A] - Size: {}, Blocks: {}", size, blocks);
                let res_node = Rc::new(RefCell::new(Node::default()));
                while blocks > 0 {
                    let res_children_node = self.parse_block()?;
                    res_node.borrow_mut().children.push(res_children_node);
                    blocks -= 1;
                }
                Ok(res_node)
            }
            0x42 => { // Signature B
                let size = self.read_i32_packed()?;
                trace!("[Signature B] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size as usize, 0);
                self.content.read_exact(&mut buffer);
                Ok(return_wrapped_pointer(SlData::Data(buffer)))
            }
            0x43 => { // Signature C
                let value = -self.content.read_i8()?;
                trace!("[Signature C] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::Char(value)))
            }
            0x44 => { // Signature D
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature D] - Size: {}, Blocks: {}", size, blocks);
                let node = Rc::new(RefCell::new(Node::default()));
                while blocks > 0 {
                    let name = self.read_string()?;
                    trace!("[Signature D] - Name: {}", name);
                    let res_node = return_wrapped_pointer(SlData::String(name));
                    let res_children_node = self.parse_block()?;
                    res_node.borrow_mut().children.push(res_children_node);
                    node.borrow_mut().children.push(res_node);
                    blocks -= 1;
                }
                Ok(node)
            }
            0x46 => { // Signature F
                let value = self.content.read_f32::<LittleEndian>()?;
                trace!("[Signature F] - Value: {:.3}", value);
                Ok(return_wrapped_pointer(SlData::Float(value)))
            }
            0x48 => { // Signature H
                let value = -self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature H] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::Word(value)))
            }
            0x49 => { // Signature I
                let value = self.content.read_i32::<LittleEndian>()?;
                trace!("[Signature I] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::Int(value)))
            }
            0x53 => { // Signature S
                let value = self.read_string()?;
                trace!("[Signature S] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::String(value)))
            }
            0x57 => { // Signature W
                let value = self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature W] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::Word(value)))
            }
            0x59 => { // Signature Y
                let value = self.content.read_u8()?;
                trace!("[Signature Y] - Value: {}", value);
                Ok(return_wrapped_pointer(SlData::UChar(value)))
            }
            s if s < 0x20 => {
                // Independent data
                trace!("[Independent Data] - Value: {}", s);
                Ok(return_wrapped_pointer(SlData::UChar(s)))
            }
            s  if s >= 0x80 => {
                // Independent data block
                let size = s as usize & 0x7f;
                trace!("[Independent Data Block] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size, 0);
                self.content.read_exact(&mut buffer);
                Ok(return_wrapped_pointer(SlData::Data(buffer)))
            }
            _ => {
                error!("Unhandled signature = 0x{:02x}, position: {}", signature, self.content.position());
                panic!("Unhandled type {}", signature);
            }
        }
    }

    /*
    /// Parse 'HEAD" block
    #[tracing::instrument(skip(self))]
    fn parse_head(&mut self) -> Result<()> {
        let (size, mut blocks) = match_block_size_number(self.parse_block()?)?;
        debug!("HEAD: size = {size}, blocks = {blocks}");
        while blocks > 0 {
            let _data = self.parse_block();
            blocks -= 1;
        }

        Ok(())
    }

    /// Parse 'MTRL" block
    #[tracing::instrument(skip(self))]
    fn parse_material(&mut self) -> Result<()> {
        let (size, mut blocks) = match_block_size_number(self.parse_block()?)?;
        debug!("MTRL: size = {size}, blocks = {blocks}");
        while blocks > 0 {
            let _data = self.parse_block();
            blocks -= 1;
        }

        Ok(())
    }
     */
}

/// Match block size & numbers
fn match_block_size_number(data: SlData) -> Result<(i32, i32)> {
    match data {
        SlData::BlockSizeNumber(s, b) => Ok((s, b)),
        _ => Err("Cannot match SlData::BlockSizeNumber")?,
    }
}

/// Prepare return data
fn return_wrapped_pointer(data: SlData) -> Rc<RefCell<Node>> {
    Rc::new(RefCell::new(Node::new(data)))
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

