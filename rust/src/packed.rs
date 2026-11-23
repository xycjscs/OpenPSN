//! Packed Cholesky preconditioning and FP64 residual-checked PCG.
use crate::{chol, sp};

use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
    slice,
};

struct Factor {
    map: Vec<i32>,
    inv_diag: Vec<f64>,
    scale: Vec<f64>,
    values: Vec<i16>,
    deltas: Vec<u8>,
    value_ptr: Vec<u32>,
    delta_ptr: Vec<u32>,
}

impl Factor {
    fn pack(l: &chol::L, map: Vec<i32>, bits: u32) -> Result<Self, String> {
        let size = map.len();
        let mut f = Self {
            map,
            inv_diag: Vec::with_capacity(size),
            scale: Vec::with_capacity(size),
            values: Vec::with_capacity(l.li.len() - size),
            deltas: Vec::with_capacity(l.li.len() - size),
            value_ptr: Vec::with_capacity(size + 1),
            delta_ptr: Vec::with_capacity(size + 1),
        };
        let peak_code = ((1u32 << (bits - 1)) - 1) as f64;
        for k in 0..size {
            f.value_ptr.push(f.values.len() as u32);
            f.delta_ptr.push(f.deltas.len() as u32);
            let lo = l.lp[k] as usize;
            let hi = l.lp[k + 1] as usize - 1;
            f.inv_diag.push(l.lx[hi].recip());
            let max = l.lx[lo..hi].iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let scale = max / peak_code;
            f.scale.push(scale);
            if max == 0.0 {
                continue;
            }
            let mul = peak_code / max;
            let mut prev = k;
            for j in lo..hi {
                let v = (l.lx[j] * mul).round() as i16;
                if v == 0 {
                    continue;
                }
                let row = l.li[j] as usize;
                let mut d = row - prev;
                while d >= 128 {
                    f.deltas.push((d & 127) as u8 | 128);
                    d >>= 7;
                }
                f.deltas.push(d as u8);
                f.values.push(v);
                prev = row;
            }
        }
        f.value_ptr.push(
            u32::try_from(f.values.len())
                .map_err(|_| "packed values exceed the uint32 offset range")?,
        );
        f.delta_ptr.push(
            u32::try_from(f.deltas.len())
                .map_err(|_| "packed row deltas exceed the uint32 offset range")?,
        );
        f.values.shrink_to_fit();
        f.deltas.shrink_to_fit();
        Ok(f)
    }

    fn bytes(&self) -> usize {
        4 * self.map.capacity()
            + 8 * (self.inv_diag.capacity() + self.scale.capacity())
            + 2 * self.values.capacity()
            + self.deltas.capacity()
            + 4 * (self.value_ptr.capacity() + self.delta_ptr.capacity())
    }

    #[inline(always)]
    unsafe fn next_row(&self, at: &mut usize, row: &mut usize) {
        let b = *self.deltas.get_unchecked(*at) as usize;
        *at += 1;
        if b < 128 {
            *row += b;
        } else {
            let mut d = b & 127;
            let mut shift = 7;
            loop {
                let b = *self.deltas.get_unchecked(*at) as usize;
                *at += 1;
                d |= (b & 127) << shift;
                if b < 128 {
                    break;
                }
                shift += 7;
            }
            *row += d;
        }
    }

    unsafe fn apply(&self, rhs: &[f64], out: &mut [f64], tmp: &mut [f64]) {
        for k in 0..tmp.len() {
            tmp[k] = *rhs.get_unchecked(*self.map.get_unchecked(k) as usize);
        }
        for k in 0..tmp.len() {
            let solved = tmp[k] * self.inv_diag[k];
            tmp[k] = solved;
            let scaled = solved * self.scale[k];
            let mut row = k;
            let mut at = self.delta_ptr[k] as usize;
            let lo = self.value_ptr[k] as usize;
            let hi = self.value_ptr[k + 1] as usize;
            for &coef in self.values.get_unchecked(lo..hi) {
                self.next_row(&mut at, &mut row);
                *tmp.get_unchecked_mut(row) -= scaled * (coef as f64);
            }
        }
        for k in (0..tmp.len()).rev() {
            let mut row = k;
            let mut at = self.delta_ptr[k] as usize;
            let lo = self.value_ptr[k] as usize;
            let hi = self.value_ptr[k + 1] as usize;
            let mut sum = 0.0;
            for &coef in self.values.get_unchecked(lo..hi) {
                self.next_row(&mut at, &mut row);
                sum += *tmp.get_unchecked(row) * (coef as f64);
            }
            let solved = (tmp[k] - self.scale[k] * sum) * self.inv_diag[k];
            tmp[k] = solved;
            *out.get_unchecked_mut(*self.map.get_unchecked(k) as usize) = solved;
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn packed_factor(
    n: usize,
    nnz: usize,
    ptr: *const i32,
    index: *const i32,
    data: *const f64,
    bits: u32,
    info: *mut u64,
) -> *mut c_void {
    if n == 0 || !(8..=16).contains(&bits) {
        return std::ptr::null_mut();
    }
    let ptr = slice::from_raw_parts(ptr, n + 1);
    let index = slice::from_raw_parts(index, nnz);
    let data = slice::from_raw_parts(data, nnz);
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Factor, String> {
        sp::check_csc(n, nnz, ptr, index, data)?;
        let (l, map) = chol::analyze_and_factor(n, ptr, index, data, 1).map_err(|(e, _)| e)?;
        let f = Factor::pack(&l, map, bits)?;
        *info.add(0) = l.li.len() as u64;
        *info.add(1) = f.values.len() as u64;
        *info.add(2) = f.deltas.len() as u64;
        *info.add(3) = f.bytes() as u64;
        Ok(f)
    }));
    match result {
        Ok(Ok(f)) => Box::into_raw(Box::new(f)).cast(),
        Ok(Err(e)) => {
            eprintln!("packed factor: {e}");
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn packed_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr.cast::<Factor>()));
    }
}

struct Stencil {
    width: usize,
    code: *const u32,
    offsets: *const i32,
    vals: *const f64,
}

impl Stencil {
    #[inline]
    unsafe fn matvec(&self, x: &[f64], out: &mut [f64]) {
        if self.width == 7 {
            return self.fixed_matvec::<7>(x, out);
        }
        for (row, out) in out.iter_mut().enumerate() {
            let begin = *self.code.add(row) as usize * self.width;
            let mut total = 0.0;
            for k in begin..begin + self.width {
                let col = row.wrapping_add_signed(*self.offsets.add(k) as isize);
                total += *self.vals.add(k) * *x.get_unchecked(col);
            }
            *out = total;
        }
    }

    #[inline]
    unsafe fn fixed_matvec<const WIDTH: usize>(&self, x: &[f64], out: &mut [f64]) {
        for (row, out) in out.iter_mut().enumerate() {
            let begin = *self.code.add(row) as usize * WIDTH;
            let mut total = 0.0;
            for k in begin..begin + WIDTH {
                let col = row.wrapping_add_signed(*self.offsets.add(k) as isize);
                total += *self.vals.add(k) * *x.get_unchecked(col);
            }
            *out = total;
        }
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut sums = [0.0; 4];
    let mut xc = x.chunks_exact(4);
    let mut yc = y.chunks_exact(4);
    for (a, b) in xc.by_ref().zip(yc.by_ref()) {
        for k in 0..4 {
            sums[k] += a[k] * b[k];
        }
    }
    sums.iter().sum::<f64>()
        + xc.remainder()
            .iter()
            .zip(yc.remainder())
            .map(|(a, b)| a * b)
            .sum::<f64>()
}

#[no_mangle]
pub unsafe extern "C" fn packed_solve(
    n: usize,
    width: usize,
    code: *const u32,
    offsets: *const i32,
    vals: *const f64,
    ptr: *const c_void,
    rhs: *const f64,
    result: *mut f64,
    relative: f64,
    limit: i32,
    achieved: *mut f64,
) -> i32 {
    let f = &*ptr.cast::<Factor>();
    let m = Stencil {
        width,
        code,
        offsets,
        vals,
    };
    let b = slice::from_raw_parts(rhs, n);
    let x = slice::from_raw_parts_mut(result, n);
    let norm = dot(b, b).sqrt();
    x.fill(0.0);
    if norm == 0.0 {
        *achieved = 0.0;
        return 0;
    }
    let target = norm * relative;
    let mut r = b.to_vec();
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut work = vec![0.0; n];
    let mut previous = 0.0;
    for step in 1..=limit {
        f.apply(&r, &mut z, &mut work);
        let rz = dot(&r, &z);
        let beta = if previous == 0.0 { 0.0 } else { rz / previous };
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        m.matvec(&p, &mut q);
        let pq = dot(&p, &q);
        if !pq.is_finite() || pq <= 0.0 {
            *achieved = dot(&r, &r).sqrt() / norm;
            return -step;
        }
        let alpha = rz / pq;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        if dot(&r, &r).sqrt() <= target || step == limit {
            m.matvec(x, &mut q);
            for i in 0..n {
                r[i] = b[i] - q[i];
            }
            *achieved = dot(&r, &r).sqrt() / norm;
            if *achieved <= relative {
                return step;
            }
            previous = 0.0;
        } else {
            previous = rz;
        }
    }
    -limit
}
