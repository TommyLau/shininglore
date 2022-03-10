use std::io::{self, Cursor, Read};
use byteorder::ReadBytesExt;

#[derive(Default)]
/// A BWX class to handle ShiningLore BNX / PNX file
pub struct BWX {
    data: Cursor<Vec<u8>>,
    size: i32,
    blocks: i32,
    pub pointer: usize,
}

impl BWX {
    /// Returns a BWX with the given file name
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice that holds the file name of a BNX / PNX file
    ///
    /// # Examples
    ///
    /// ```
    /// use bwx::BWX;
    /// let bwx = BWX::new();
    /// ```
    pub fn new() -> Self {
        BWX { ..Default::default() }
    }

    pub fn load_from_file(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("Read from file: {filename}");

        let data = std::fs::read(filename)?;
        self.data = Cursor::new(data);
        //println!("{:?}", data);

        self.check_bwx_header()?;
        self.data.set_position(4);

        self.size = self.read_i32_packed()?;
        self.blocks = self.read_i32_packed()?;
        println!("sz: {}, blocks: {}", self.size, self.blocks);

        let name = self.read_string()?;
        println!("section name: {name}");

        println!("Pointer: {}", self.data.position());

        Ok(())
    }

    /// Check whether the file is a valid BNX / PNX format
    fn check_bwx_header(&mut self) -> Result<(), String> {
        let header = &self.data.get_ref()[..4];
        if header != "BWXF".as_bytes() {
            return Err("Invalid BWX file.".to_string());
        }

        Ok(())
    }

    /// Read BNX / PNX special packed integer value (little endian)
    fn read_i32_packed(&mut self) -> Result<i32, io::Error> {
        let mut result: u32 = 0;
        let mut shift = 0;

        while shift < 35 {
            let t = self.data.read_u8()? as u32;
            result |= (t & 0x7f) << shift;

            if t & 0x80 == 0 {
                break;
            } else {
                shift += 7;
            }
        }

        Ok(result as i32)
    }

    /// Read string
    fn read_string(&mut self) -> Result<String, io::Error> {
        let length = self.data.read_u8()?;
        let mut buffer = Vec::new();
        buffer.resize(length as usize, 0);
        self.data.read_exact(&mut buffer)?;

        Ok(String::from_utf8_lossy(&buffer).trim_matches('\0').to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_bwx_header() {
        let mut bwx = BWX { ..Default::default() };
        bwx.data = Cursor::new(vec![66, 87, 88, 70]);
        assert!(bwx.check_bwx_header().is_ok(), "File header check should pass");
        bwx.data = Cursor::new(vec![11, 22, 33, 44]);
        assert!(bwx.check_bwx_header().is_err(), "File header check should fail");
    }

    #[test]
    fn read_i32_packed() {
        let mut bwx = BWX { ..Default::default() };
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.data = Cursor::new(vec![0xc1, 0xef, 0x5a, 0x0c]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 1488833, "Packed integer value incorrect");
        bwx.data = Cursor::new(vec![0x0c, 0x02]);
        assert_eq!(bwx.read_i32_packed().unwrap(), 12, "Packed integer value incorrect");
    }

    #[test]
    fn read_string() {
        let mut bwx = BWX { ..Default::default() };
        // Data from "EXTERNAL_UI_DEFAULT.PNX"
        bwx.data = Cursor::new(vec![0x02, 0x30, 0x00, 0x53]);
        assert_eq!(bwx.read_string().unwrap().as_str(), "0", "The string should be '0'");
    }
}
