pub unsafe fn sneaky() {}
pub trait CAbi { extern "C" fn required(); }
unsafe extern "C" {
    pub safe fn foreign_safe();
    pub fn foreign_implicit();
    pub unsafe fn foreign_unsafe();
    pub safe static SAFE_STATIC: u8;
    pub static IMPLICIT_STATIC: u8;
}
pub static PLAIN: u8 = 1;
pub unsafe trait UnsafeTrait {}
