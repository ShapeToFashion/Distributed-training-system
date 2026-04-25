package proto

import "math"

// PackInt8 quantizes gradients to int8 and packs 4 per float32.
// Format: [scale][origLen][packed_int8...]
// 4x smaller than float32, 2x smaller than float16.
func PackInt8(src []float32) []float32 {
	if len(src) == 0 {
		return nil
	}
	maxAbs := float32(0)
	for _, v := range src {
		if a := abs32(v); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		maxAbs = 1
	}
	scale := maxAbs / 127.0
	nPacked := (len(src) + 3) / 4
	result := make([]float32, nPacked+2)
	result[0] = scale
	result[1] = math.Float32frombits(uint32(len(src)))
	for i := 0; i < nPacked; i++ {
		var b [4]uint8
		for j := 0; j < 4; j++ {
			if i*4+j < len(src) {
				qi := int(src[i*4+j] / scale)
				if qi > 127 {
					qi = 127
				}
				if qi < -128 {
					qi = -128
				}
				b[j] = uint8(int8(qi))
			}
		}
		result[i+2] = math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	}
	return result
}

// UnpackInt8 reverses PackInt8.
func UnpackInt8(data []float32) []float32 {
	if len(data) < 2 {
		return nil
	}
	scale := data[0]
	origLen := int(math.Float32bits(data[1]))
	result := make([]float32, origLen)
	for i, p := range data[2:] {
		bits := math.Float32bits(p)
		for j := 0; j < 4; j++ {
			if i*4+j < origLen {
				result[i*4+j] = float32(int8((bits>>(j*8))&0xFF)) * scale
			}
		}
	}
	return result
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// PackF16 converts float32 weights/gradients to float16 and packs two float16
// values into each float32 slot — halving the array length before sending.
// The original length is stored in the first element so the receiver can
// reconstruct the exact slice without extra fields.
func PackF16(src []float32) []float32 {
	packed := make([]float32, (len(src)+1)/2)
	for i := range packed {
		hi := f32ToF16(src[i*2])
		lo := uint16(0)
		if i*2+1 < len(src) {
			lo = f32ToF16(src[i*2+1])
		}
		packed[i] = math.Float32frombits((uint32(hi) << 16) | uint32(lo))
	}
	// prepend original length as first element
	result := make([]float32, len(packed)+1)
	result[0] = math.Float32frombits(uint32(len(src)))
	copy(result[1:], packed)
	return result
}

// UnpackF16 reverses PackF16.
func UnpackF16(data []float32) []float32 {
	if len(data) == 0 {
		return nil
	}
	origLen := int(math.Float32bits(data[0]))
	packed := data[1:]
	result := make([]float32, origLen)
	for i := range packed {
		bits := math.Float32bits(packed[i])
		hi := uint16(bits >> 16)
		lo := uint16(bits & 0xFFFF)
		if i*2 < origLen {
			result[i*2] = f16ToF32(hi)
		}
		if i*2+1 < origLen {
			result[i*2+1] = f16ToF32(lo)
		}
	}
	return result
}

func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 31) & 0x1)
	exp := int32((b>>23)&0xFF) - 127 + 15
	mant := b & 0x7FFFFF
	if exp <= 0 {
		return sign << 15
	}
	if exp >= 31 {
		return (sign << 15) | 0x7C00
	}
	return (sign << 15) | uint16(exp<<10) | uint16(mant>>13)
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3FF
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7FC00000)
	}
	return math.Float32frombits((sign << 31) | ((exp+127-15)<<23) | (mant << 13))
}
