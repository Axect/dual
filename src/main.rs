use std::ops::{Neg, Add, Sub, Mul, Div};

#[derive(Debug, Copy, Clone)]
struct Dual {
    x: f64,
    dx: f64,
}

trait Ops {
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn powi(self, n: i32) -> Self;
}

trait Sigmoid: Sized 
    + Ops
    + Neg<Output=Self>
    + Add<f64, Output=Self> 
where
    f64: Div<Self, Output=Self> {
    fn sigmoid(self) -> Self {
        1f64 / ((-self).exp() + 1f64)
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            dx: self.dx + rhs.dx,
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            dx: self.dx - rhs.dx,
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.x * rhs.x,
            dx: self.x * rhs.dx + self.dx * rhs.x,
        }
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            x: self.x / rhs.x,
            dx: (self.dx * rhs.x - self.x * rhs.dx) / (rhs.x * rhs.x),
        }
    }
}

impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            x: self.x + rhs,
            dx: self.dx,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            x: self.x - rhs,
            dx: self.dx,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            dx: self.dx * rhs,
        }
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            x: self.x / rhs,
            dx: self.dx / rhs,
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;

    fn div(self, rhs: Dual) -> Dual {
        Dual {
            x: self / rhs.x,
            dx: -(self * rhs.dx) / (rhs.x * rhs.x),
        }
    }
}

impl Ops for Dual {
    fn exp(self) -> Self {
        Self {
            x: self.x.exp(),
            dx: self.x * self.dx.exp(),
        }
    }

    fn ln(self) -> Self {
        Self {
            x: self.x.ln(),
            dx: self.dx / self.x,
        }
    }

    fn sin(self) -> Self {
        Self {
            x: self.x.sin(),
            dx: self.x.cos() * self.dx,
        }
    }

    fn cos(self) -> Self {
        Self {
            x: self.x.cos(),
            dx: -self.x.sin() * self.dx,
        }
    }

    fn tan(self) -> Self {
        let tan = self.x.tan();
        Self {
            x: tan,
            dx: self.dx * (tan * tan + 1.0),
        }
    }

    fn powi(self, n: i32) -> Self {
        Self {
            x: self.x.powi(n),
            dx: n as f64 * self.x.powi(n - 1) * self.dx,
        }
    }
}

impl Sigmoid for Dual {}

fn main() {
    let u = Dual { x: 1.0, dx: 1.0 }; // x at x=1
    let v = u.sin();
    println!("v: {}, dv: {}", v.x, v.dx);
    // sin(1), sin'(x) = cos(x) = cos(1)

    let w = v.sin();
    println!("w: {}, dw: {}", w.x, w.dx);
    // sin(sin(1))=sin(1), sin'(x^2) = 2*x*cos(x^2) = 2cos(1)

    let z = u.sigmoid();
    println!("z: {}, dz: {}", z.x, z.dx);
    // sigmoid(1), sigmoid'(x) = sigmoid(1) * (1 - sigmoid(1))
}
