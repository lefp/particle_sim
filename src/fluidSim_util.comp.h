/// Get the index of the cell that contains the particle.
uvec3 cellIndex(vec3 particle, vec3 domain_min, float cell_size_reciprocal) {

    return uvec3((particle - domain_min) * cell_size_reciprocal);
}

// For some input integer with bits [... b3 b2 b1 b0]
// returns [... 0 0 b3 0 0 b2 0 0 b1 0 0 b0].
uint separateBitsByTwo(uint x) {
    // https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
    // TODO FIXME OPTIMIZE there's a faster way to do this:
    //     https://fgiesen.wordpress.com/2022/09/09/morton-codes-addendum/
    x &= 0x000003ff;                  // 0b0000'0000'0000'0000'0000'0011'1111'1111
    x = (x ^ (x << 16)) & 0xff0000ff; // 0b1111'1111'0000'0000'0000'0000'1111'1111
    x = (x ^ (x <<  8)) & 0x0300f00f; // 0b0000'0011'0000'0000'1111'0000'0000'1111
    x = (x ^ (x <<  4)) & 0x030c30c3; // 0b0000'0011'0000'1100'0011'0000'1100'0011
    x = (x ^ (x <<  2)) & 0x09249249; // 0b0000'1001'0010'0100'1001'0010'0100'1001
    return x;
}

uint cellMortonCode(uvec3 cell_index) {
    return
        (separateBitsByTwo(cell_index.x)     ) |
        (separateBitsByTwo(cell_index.y) << 1) |
        (separateBitsByTwo(cell_index.z) << 2) ;
}

