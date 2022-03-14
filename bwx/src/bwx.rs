use std::cell::{Ref, RefCell};
use std::io::{Cursor, Read};
use std::rc::Rc;
use byteorder::{LittleEndian, ReadBytesExt};
use tracing::{debug, error, info, trace, warn};

// u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize, f32, f64
#[derive(Debug, Clone)]
pub enum SlData {
    UChar(u8),
    Char(i8),
    Word(i16),
    Int(i32),
    Float(f32),
    String(String),
    Data(Vec<u8>),
    Array,
    DArray(String),
    None,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub data: SlData,
    pub children: Option<Vec<Rc<RefCell<Node>>>>,
}

#[derive(Debug, Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    content: Cursor<Vec<u8>>,
    pub node: Node,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

impl Default for Node {
    fn default() -> Self {
        Node {
            data: SlData::None,
            children: None,
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

    /// Find a block with the given name
    ///
    /// # Arguments
    ///
    /// * name - The name of the block
    ///
    /// # Examples
    ///
    /// ```
    /// use bwx::{Node, SlData};
    /// let node = Node::new(SlData::String("root".into()));
    /// let res_node = node.find_block("root");
    /// ```
    #[tracing::instrument(skip(self))]
    pub fn find_block(&self, name: &str) -> Node {
        /*
        for child in self.children.iter() {
            match &child.borrow().data {
                SlData::String(string) => {
                    if string == name {
                        debug!("Found block: [{}]", string);
                        return self.clone();
                    } else {
                        let res_node = child.borrow().find_block(name);
                        if let SlData::None = &res_node.data {} else { return res_node; }
                    }
                }
                SlData::Array => {
                    let res_node = child.borrow().find_block(name);
                    debug!("Array: {:?}", res_node);
                    if let SlData::None = &res_node.data {} else { return res_node; }
                }
                SlData::DArray(string) => {
                    return if string == name {
                        debug!("Found D-block: [{}]", string);
                        child.borrow().clone()
                    } else {
                        child.borrow().find_block(name)
                    };
                }
                _ => {}
            }
        }

         */
        Node::new(SlData::None)
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
        BWX {
            node: Node {
                data: SlData::String("root".into()),
                children: Some(vec![]),
            },
            ..Default::default()
        }
    }

    #[tracing::instrument(skip(self, filename))]
    pub fn load_from_file(&mut self, filename: &str) -> Result<()> {
        info!(filename);

        let data = std::fs::read(filename)?;
        self.content = Cursor::new(data);

        self.check_bwx_header()?;
        self.content.set_position(4);

        let (size, mut blocks) = self.read_block_size_number()?;
        trace!(size, blocks);

        while blocks > 0 {
            let name = self.read_string()?;
            trace!("Main block name: {}", name);
            //let node = return_wrapped_pointer(SlData::String(name));
            let node = Rc::new(RefCell::new(Node {
                data: SlData::String(name),
                children: Some(vec![]),
            }));
            self.go_through(&node)?;
            //self.node.children.push(node);
            self.node.children.as_mut().unwrap().push(node);
            blocks -= 1;
        }

        // Parse data and get mesh data
        self.parse_mesh()?;

        Ok(())
    }


    ///
    fn parse_mesh(&self) -> Result<()> {
        /*
        let name = match &node.data {
            SlData::String(s) => s,
            _ => "",
        };
        debug!("{:#?}",name);
        debug!("{:#?}",node.children.len());

         */
        let node = self.node.find_block("0");
        //debug!("{:#?}", node.data);
        let slbwx = node.find_block("SLBWX");
        let spob = self.node.find_block("SPOB");
        let dxobj = self.node.find_block("DXOBJ");
        debug!("{:#?}", self.node);
        panic!("Debug");

        /*
        let mut data = Vec::new();

        for child in self.node.children.iter() {
            debug!("{:?}", child.borrow_mut().data);
            if let SlData::String(s) = &child.borrow_mut().data {
                match s.as_str() {
                    "DXOBJ" | "SPOB" => {
                        debug!("{:?}", s);
                        data.push(child.clone());
                    }
                    _ => {}
                }
            }
        }

        // working ok
        for d in data.iter() {
            debug!("len: {}", d.borrow_mut().children.len());
            /*
            debug!("{:#?}", d.borrow_mut().data);
            debug!("len: {}", d.borrow_mut().children.len());
            debug!("{:#?}", d.borrow_mut().children);
             */
            //let e = d.borrow_mut().children.get(0).unwrap().clone();
            //debug!("len: {}", e.borrow_mut().children.len());
        }

        //debug!("{:#?}", data);


         */
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
    //#[tracing::instrument(skip(self))]
    //fn go_through(&mut self, node: &Rc<RefCell<Node>>) -> Result<Rc<RefCell<Node>>> {
    #[tracing::instrument(skip(self, node))]
    fn go_through(&mut self, node: &Rc<RefCell<Node>>) -> Result<()> {
        let signature = self.content.read_u8()?;

        let data = match signature {
            0x41 => { // Signature A
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature A] - Size: {}, Blocks: {}", size, blocks);
                //let res_node = return_wrapped_pointer(SlData::Array);
                let res_node = Rc::new(RefCell::new(Node {
                    data: SlData::Array,
                    children: Some(vec![]),
                }));
                while blocks > 0 {
                    self.go_through(&res_node)?;
                    //res_node.borrow_mut().children.push(res_children_node);
                    blocks -= 1;
                }
                node.borrow_mut().children.as_mut().unwrap().push(res_node);
                return Ok(());
            }
            0x42 => { // Signature B
                let size = self.read_i32_packed()?;
                trace!("[Signature B] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size as usize, 0);
                self.content.read_exact(&mut buffer)?;
                SlData::Data(buffer)
            }
            0x43 => { // Signature C
                let value = -self.content.read_i8()?;
                trace!("[Signature C] - Value: {}", value);
                SlData::Char(value)
            }
            0x44 => { // Signature D
                let (size, mut blocks) = self.read_block_size_number()?;
                trace!("[Signature D] - Size: {}, Blocks: {}", size, blocks);
                while blocks > 0 {
                    let name = self.read_string()?;
                    trace!("[Signature D] - Name: {}", name);
                    let res_node = Rc::new(RefCell::new(Node {
                        data: SlData::DArray(name),
                        children: Some(vec![]),
                    }));
                    self.go_through(&res_node)?;
                    node.borrow_mut().children.as_mut().unwrap().push(res_node);
                    blocks -= 1;
                }
                return Ok(());
            }
            0x46 => { // Signature F
                let value = self.content.read_f32::<LittleEndian>()?;
                trace!("[Signature F] - Value: {:.3}", value);
                SlData::Float(value)
            }
            0x48 => { // Signature H
                let value = -self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature H] - Value: {}", value);
                SlData::Word(value)
            }
            0x49 => { // Signature I
                let value = self.content.read_i32::<LittleEndian>()?;
                trace!("[Signature I] - Value: {}", value);
                SlData::Int(value)
            }
            0x53 => { // Signature S
                let value = self.read_string()?;
                trace!("[Signature S] - Value: {}", value);
                SlData::String(value)
            }
            0x57 => { // Signature W
                let value = self.content.read_i16::<LittleEndian>()?;
                trace!("[Signature W] - Value: {}", value);
                SlData::Word(value)
            }
            0x59 => { // Signature Y
                let value = self.content.read_u8()?;
                trace!("[Signature Y] - Value: {}", value);
                SlData::UChar(value)
            }
            s if s < 0x20 => {
                // Independent data
                trace!("[Independent Data] - Value: {}", s);
                SlData::UChar(s)
            }
            s  if s >= 0x80 => {
                // Independent data block
                let size = s as usize & 0x7f;
                trace!("[Independent Data Block] - Size: {}", size);
                let mut buffer: Vec<u8> = Vec::new();
                buffer.resize(size, 0);
                self.content.read_exact(&mut buffer)?;
                SlData::Data(buffer)
            }
            _ => {
                error!("Unhandled signature = 0x{:02x}, position: {}", signature, self.content.position());
                //debug!("{:#?}", self.node);
                panic!("Unhandled type {}", signature);
            }
        };

        node.borrow_mut().children.as_mut().unwrap().push(return_wrapped_pointer(data));
        Ok(())
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

