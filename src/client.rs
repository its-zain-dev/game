  #![cfg(target_arch = "wasm32")]
  //! العميل: الرسم + الشبكة + كيبورد/ماوس/لمس/جويستك + شات + ملء الشاشة. كله Rust

  use std::cell::RefCell;
  use std::collections::{HashMap, VecDeque};
  use std::f32::consts::{PI, TAU};
  use std::rc::Rc;
  use wasm_bindgen::prelude::*;
  use wasm_bindgen::JsCast;
  use web_sys::{
      BinaryType, Document, Element, Event, EventTarget, Gamepad, GamepadButton, HtmlCanvasElement,
      HtmlElement, HtmlInputElement, KeyboardEvent, MessageEvent, MouseEvent, Navigator,
      OrientationLockType, PointerEvent, WebGl2RenderingContext as GL, WebGlBuffer, WebGlProgram,
      WebGlUniformLocation, WebGlVertexArrayObject, WebSocket, WheelEvent, Window,
  };

  const WALK: f32 = 5.0;
  const RUN: f32 = 10.0;
  const GRAVITY: f32 = 25.0;
  const JUMP: f32 = 9.0;
  const SEND_MS: f64 = 16.0;
  const KEEPALIVE_MS: f64 = 5000.0;
  const LABEL_DIST: f32 = 60.0;
  const BUBBLE_MS: f64 = 5000.0;
  const CHAT_LINE_MS: f64 = 15000.0;
  const CHAT_LINES: u32 = 8;
  const SKY: [f32; 3] = [0.86, 0.90, 0.95];
  const DEAD: f32 = 0.15;
  const JOY_R: f32 = 60.0;

  thread_local! {
      static NOOP: Closure<dyn FnMut(JsValue)> = Closure::new(|_: JsValue| {});
  }

  // ---------------- Shaders ----------------
  const GROUND_VS: &str = r#"#version 300 es
  uniform mat4 vp;uniform vec2 off;out vec2 lp;out vec2 gp;
  void main(){vec2 c=vec2(gl_VertexID&1,gl_VertexID>>1)*2.-1.;lp=c*1500.;gp=lp+off;gl_Position=vp*vec4(lp.x,0.,lp.y,1.);}"#;

  const GROUND_FS: &str = r#"#version 300 es
  precision highp float;in vec2 lp;in vec2 gp;out vec4 o;uniform vec3 sky;
  vec3 fog(vec3 c,float d){return mix(c,sky,1.-exp(-d*d*.000004));}
  float grid(float s){vec2 q=gp/s;vec2 g=abs(fract(q-.5)-.5)/fwidth(q);return 1.-min(min(g.x,g.y),1.);}
  void main(){vec3 c=vec3(.97)-vec3(.05)*grid(1.)-vec3(.1)*grid(10.);o=vec4(fog(c,length(lp)),1.);}"#;

  const BOX_VS: &str = r#"#version 300 es
  layout(location=0)in vec3 p;layout(location=1)in vec3 n;layout(location=2)in vec4 ip;layout(location=3)in vec3 sc;layout(location=4)in vec3 cl;
  uniform mat4 vp;out vec3 vn;out vec3 vc;out vec3 wp;
  void main(){float c=cos(ip.w),s=sin(ip.w);vec3 q=p*sc;q=vec3(c*q.x+s*q.z,q.y,-s*q.x+c*q.z)+ip.xyz;
  vn=vec3(c*n.x+s*n.z,n.y,-s*n.x+c*n.z);vc=cl;wp=q;gl_Position=vp*vec4(q,1.);}"#;

  const BOX_FS: &str = r#"#version 300 es
  precision highp float;in vec3 vn;in vec3 vc;in vec3 wp;out vec4 o;uniform vec3 sky;
  vec3 fog(vec3 c,float d){return mix(c,sky,1.-exp(-d*d*.000004));}
  void main(){vec3 N=normalize(vn);float d=max(dot(N,normalize(vec3(.4,1.,.3))),0.);float h=.5+.5*N.y;
  o=vec4(fog(vc*(.35+.15*h+.5*d),length(wp.xz)),1.);}"#;

  const SHADOW_VS: &str = r#"#version 300 es
  layout(location=0)in vec3 s;uniform mat4 vp;out vec2 uv;
  void main(){vec2 c=vec2(gl_VertexID&1,gl_VertexID>>1)*2.-1.;uv=c;gl_Position=vp*vec4(s.x+c.x*s.z,.01,s.y+c.y*s.z,1.);}"#;

  const SHADOW_FS: &str = r#"#version 300 es
  precision mediump float;in vec2 uv;out vec4 o;
  void main(){o=vec4(0.,0.,0.,smoothstep(1.,.25,length(uv))*.3);}"#;

  // ---------------- Renderer: 3 draw calls بس ----------------
  struct Renderer {
      gp: WebGlProgram, g_vp: Option<WebGlUniformLocation>, g_off: Option<WebGlUniformLocation>,
      g_sky: Option<WebGlUniformLocation>, g_vao: WebGlVertexArrayObject,
      bp: WebGlProgram, b_vp: Option<WebGlUniformLocation>, b_sky: Option<WebGlUniformLocation>,
      b_vao: WebGlVertexArrayObject, ib: WebGlBuffer,
      sp: WebGlProgram, s_vp: Option<WebGlUniformLocation>, s_vao: WebGlVertexArrayObject, sb: WebGlBuffer,
      inst: Vec<f32>,
      shad: Vec<f32>,
  }

  fn bytes(v: &[f32]) -> &[u8] {
      unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
  }

  fn shader(gl: &GL, t: u32, src: &str) -> Result<web_sys::WebGlShader, JsValue> {
      let s = gl.create_shader(t).ok_or("shader")?;
      gl.shader_source(&s, src);
      gl.compile_shader(&s);
      if gl.get_shader_parameter(&s, GL::COMPILE_STATUS).as_bool() == Some(true) { Ok(s) }
      else { Err(gl.get_shader_info_log(&s).unwrap_or_default().into()) }
  }

  fn program(gl: &GL, vs: &str, fs: &str) -> Result<WebGlProgram, JsValue> {
      let p = gl.create_program().ok_or("program")?;
      gl.attach_shader(&p, &shader(gl, GL::VERTEX_SHADER, vs)?);
      gl.attach_shader(&p, &shader(gl, GL::FRAGMENT_SHADER, fs)?);
      gl.link_program(&p);
      if gl.get_program_parameter(&p, GL::LINK_STATUS).as_bool() == Some(true) { Ok(p) }
      else { Err(gl.get_program_info_log(&p).unwrap_or_default().into()) }
  }

  fn attrib(gl: &GL, loc: u32, size: i32, stride: i32, off: i32, div: u32) {
      gl.enable_vertex_attrib_array(loc);
      gl.vertex_attrib_pointer_with_i32(loc, size, GL::FLOAT, false, stride, off);
      if div > 0 { gl.vertex_attrib_divisor(loc, div); }
  }

  fn cube() -> Vec<f32> {
      let mut v = Vec::with_capacity(216);
      for n in [[1.0f32, 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]] {
          let u = [n[1], n[2], n[0]];
          let w = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
          for (a, b) in [(-1.0f32, -1.0f32), (1., -1.), (1., 1.), (-1., -1.), (1., 1.), (-1., 1.)] {
              for i in 0..3 { v.push((n[i] + u[i] * a + w[i] * b) * 0.5); }
              v.extend_from_slice(&n);
          }
      }
      v
  }

  impl Renderer {
      fn new(gl: &GL) -> Result<Self, JsValue> {
          let gp = program(gl, GROUND_VS, GROUND_FS)?;
          let bp = program(gl, BOX_VS, BOX_FS)?;
          let sp = program(gl, SHADOW_VS, SHADOW_FS)?;
          let g_vao = gl.create_vertex_array().ok_or("vao")?;

          let b_vao = gl.create_vertex_array().ok_or("vao")?;
          gl.bind_vertex_array(Some(&b_vao));
          let vb = gl.create_buffer().ok_or("buf")?;
          gl.bind_buffer(GL::ARRAY_BUFFER, Some(&vb));
          gl.buffer_data_with_u8_array(GL::ARRAY_BUFFER, bytes(&cube()), GL::STATIC_DRAW);
          attrib(gl, 0, 3, 24, 0, 0);
          attrib(gl, 1, 3, 24, 12, 0);
          let ib = gl.create_buffer().ok_or("buf")?;
          gl.bind_buffer(GL::ARRAY_BUFFER, Some(&ib));
          attrib(gl, 2, 4, 40, 0, 1);
          attrib(gl, 3, 3, 40, 16, 1);
          attrib(gl, 4, 3, 40, 28, 1);

          let s_vao = gl.create_vertex_array().ok_or("vao")?;
          gl.bind_vertex_array(Some(&s_vao));
          let sb = gl.create_buffer().ok_or("buf")?;
          gl.bind_buffer(GL::ARRAY_BUFFER, Some(&sb));
          attrib(gl, 0, 3, 12, 0, 1);
          gl.bind_vertex_array(None);

          gl.enable(GL::DEPTH_TEST);
          gl.blend_func(GL::SRC_ALPHA, GL::ONE_MINUS_SRC_ALPHA);
          Ok(Self {
              g_vp: gl.get_uniform_location(&gp, "vp"), g_off: gl.get_uniform_location(&gp, "off"),
              g_sky: gl.get_uniform_location(&gp, "sky"), gp, g_vao,
              b_vp: gl.get_uniform_location(&bp, "vp"), b_sky: gl.get_uniform_location(&bp, "sky"), bp, b_vao, ib,
              s_vp: gl.get_uniform_location(&sp, "vp"), sp, s_vao, sb,
              inst: Vec::with_capacity(10 * 6 * 64), shad: Vec::with_capacity(3 * 64),
          })
      }

          fn elytra(&mut self, x: f32, y: f32, z: f32, yw: f32, c: [f32; 3], t: f32, roll: f32, boost: bool) {
        let (sn, cs) = yw.sin_cos();
        let (fx, fz, rx, rz) = (-sn, -cs, cs, -sn);
        let at = |f: f32, r: f32| (x + fx * f + rx * r, z + fz * f + rz * r);
        let dark = c.map(|v| v * 0.55);
        let mid = c.map(|v| v * 0.75);
        let by = y + 0.55;
        let (qx, qz) = at(0.0, 0.0);
        self.bx([qx, by, qz, yw], [0.62, 0.36, 0.72], c);
        let (qx, qz) = at(0.58, 0.0);
        self.bx([qx, by + 0.03, qz, yw], [0.46, 0.44, 0.44], c.map(|v| v * 0.8 + 0.2));
        let (qx, qz) = at(0.81, 0.0);
        self.bx([qx, by + 0.08, qz, yw], [0.36, 0.13, 0.04], [0.08, 0.1, 0.14]);
        for &k in [-1.0f32, 1.0].iter() {
            let kick = (t * 8.0 + k).sin() * 0.035;
            let (qx, qz) = at(-0.72, 0.14 * k);
            self.bx([qx, by + kick, qz, yw], [0.22, 0.24, 0.72], dark);
            let (qx, qz) = at(0.0, 0.4 * k);
            self.bx([qx, by - 0.04, qz, yw], [0.16, 0.18, 0.64], mid);
            for i in 0..7 {
                let fi = i as f32;
                let span = 0.22 + fi * 0.16;
                let chord = (0.95 - fi * 0.1).max(0.32);
                let back = 0.22 - fi * 0.12 - chord * 0.5;
                let flut = (t * 14.0 + fi * 0.8).sin() * 0.006 * fi;
                let dy = 0.22 - fi * 0.03 + k * roll * span * 0.6 + flut;
                let col = if i >= 5 { [0.34, 0.31, 0.44] } else { [0.6, 0.58, 0.7] };
                let (qx, qz) = at(back, span * k);
                self.bx([qx, by + dy, qz, yw], [0.17, 0.045, chord], col);
            }
        }
        if boost {
            let fl = 0.22 + (t * 40.0).sin().abs() * 0.14;
            let (qx, qz) = at(-1.2, 0.0);
            self.bx([qx, by, qz, yw], [fl, fl, fl * 1.8], [1.0, 0.62, 0.18]);
        }
        self.shad.extend_from_slice(&[x, z, 0.7 * (1.0 - (y * 0.02).min(0.7))]);
    }

    fn folded(&mut self, x: f32, y: f32, z: f32, yw: f32, ph: f32, amp: f32) {
        let (sn, cs) = yw.sin_cos();
        let (fx, fz, rx, rz) = (-sn, -cs, cs, -sn);
        let bob = ph.sin().abs() * 0.06 * amp;
        for &k in [-1.0f32, 1.0].iter() {
            self.bx([x - fx * 0.23 + rx * 0.15 * k, y + 1.0 + bob, z - fz * 0.23 + rz * 0.15 * k, yw], [0.28, 0.8, 0.06], [0.58, 0.56, 0.68]);
            self.bx([x - fx * 0.24 + rx * 0.2 * k, y + 0.58 + bob, z - fz * 0.24 + rz * 0.2 * k, yw], [0.18, 0.12, 0.06], [0.34, 0.31, 0.44]);
        }
    }

    fn weapon(&mut self, x: f32, y: f32, z: f32, yw: f32, w: u8, flying: bool, swing: f32) {
        if w == 0 { return; }
        let (sn, cs) = yw.sin_cos();
        let (fx, fz, rx, rz) = (-sn, -cs, cs, -sn);
        let at = |f: f32, r: f32| (x + fx * f + rx * r, z + fz * f + rz * r);
        if w == 2 {
            let gy = if flying { y + 0.52 } else { y + 0.82 };
            let kick = if swing < 1.0 { (1.0 - swing) * 0.08 } else { 0.0 };
            let (qx, qz) = at(if flying { 0.45 } else { 0.3 } - kick, 0.42);
            self.bx([qx, gy, qz, yw], [0.08, 0.12, 0.42], [0.14, 0.14, 0.16]);
            let (qx, qz) = at(if flying { 0.33 } else { 0.18 }, 0.42);
            self.bx([qx, gy - 0.1, qz, yw], [0.07, 0.16, 0.09], [0.28, 0.2, 0.13]);
            if swing < 0.35 {
                let (qx, qz) = at(if flying { 0.76 } else { 0.6 }, 0.42);
                self.bx([qx, gy + 0.01, qz, yw], [0.16, 0.16, 0.16], [1.0, 0.85, 0.35]);
            }
        } else {
            let wood = [0.58, 0.38, 0.2];
            if flying {
                let (qx, qz) = at(0.0, 0.52);
                self.bx([qx, y + 0.58, qz, yw], [0.09, 0.09, 0.85], wood);
            } else if swing < 1.0 {
                let (qx, qz) = at(0.55, 0.45 - swing * 0.9);
                self.bx([qx, y + 1.05, qz, yw], [0.11, 0.11, 0.9], wood);
            } else {
                let (qx, qz) = at(0.12, 0.43);
                self.bx([qx, y + 1.12, qz, yw], [0.09, 0.8, 0.09], wood);
            }
        }
    }

    fn bar(&mut self, x: f32, y: f32, z: f32, yw: f32, hp: u8) {
        let (sn, cs) = yw.sin_cos();
        let w = hp.min(100) as f32 / 100.0;
        let (rx, rz) = (cs, -sn);
        let (tx, tz) = (sn * 0.02, cs * 0.02);
        self.bx([x, y, z, yw], [0.9, 0.1, 0.02], [0.1, 0.1, 0.1]);
        let off = -(1.0 - w) * 0.43;
        self.bx([x + rx * off + tx, y, z + rz * off + tz, yw], [0.86 * w, 0.07, 0.02], [1.0 - w, 0.25 + w * 0.65, 0.2]);
    }

    fn tracer(&mut self, s: [f32; 3], e: [f32; 3], fade: f32) {
        let d = [e[0] - s[0], e[1] - s[1], e[2] - s[2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let n = ((l / 0.8) as usize).clamp(1, 190);
        let yw = (-d[0]).atan2(-d[2]);
        let sz = 0.04 + 0.04 * fade;
        for i in 0..=n {
            let k = i as f32 / n as f32;
            self.bx([s[0] + d[0] * k, s[1] + d[1] * k, s[2] + d[2] * k, yw], [sz, sz, 0.7], [1.0, 0.92, 0.5]);
        }
        let im = 0.08 + 0.2 * fade;
        self.bx([e[0], e[1], e[2], yw], [im, im, im], [1.0, 0.6, 0.3]);
    }
    fn begin(&mut self) { self.inst.clear(); self.shad.clear(); }

      #[inline]
      fn bx(&mut self, p: [f32; 4], s: [f32; 3], c: [f32; 3]) {
          self.inst.extend_from_slice(&[p[0], p[1], p[2], p[3], s[0], s[1], s[2], c[0], c[1], c[2]]);
      }

      fn hero(&mut self, x: f32, y: f32, z: f32, yw: f32, c: [f32; 3], ph: f32, amp: f32) {
          let (sn, cs) = yw.sin_cos();
          let (fx, fz, rx, rz) = (-sn, -cs, cs, -sn);
          let s = ph.sin() * 0.28 * amp;
          let bob = ph.sin().abs() * 0.06 * amp;
          let dark = c.map(|v| v * 0.55);
          let mid = c.map(|v| v * 0.75);
          for k in [-1.0f32, 1.0] {
              let sw = s * k;
              self.bx([x + rx * 0.16 * k + fx * sw, y + 0.35 + sw.max(0.0) * 0.15, z + rz * 0.16 * k + fz * sw, yw], [0.22, 0.7, 0.26], dark);
              self.bx([x + rx * 0.41 * k - fx * sw * 0.8, y + 1.05 + bob, z + rz * 0.41 * k - fz * sw * 0.8, yw], [0.16, 0.62, 0.2], mid);
          }
          self.bx([x, y + 1.05 + bob, z, yw], [0.62, 0.72, 0.36], c);
          self.bx([x, y + 1.66 + bob, z, yw], [0.46, 0.46, 0.46], c.map(|v| v * 0.8 + 0.2));
          self.bx([x + fx * 0.22, y + 1.7 + bob, z + fz * 0.22, yw], [0.36, 0.13, 0.06], [0.08, 0.1, 0.14]);
          self.shad.extend_from_slice(&[x, z, 0.55 * (1.0 - (y * 0.2).min(0.6))]);
      }

      fn draw(&self, gl: &GL, vp: &[f32; 16], off: (f32, f32)) {
          gl.clear_color(SKY[0], SKY[1], SKY[2], 1.0);
          gl.clear(GL::COLOR_BUFFER_BIT | GL::DEPTH_BUFFER_BIT);
          // 1) الأرض
          gl.use_program(Some(&self.gp));
          gl.uniform_matrix4fv_with_f32_array(self.g_vp.as_ref(), false, vp);
          gl.uniform3f(self.g_sky.as_ref(), SKY[0], SKY[1], SKY[2]);
          gl.uniform2f(self.g_off.as_ref(), off.0, off.1);
          gl.bind_vertex_array(Some(&self.g_vao));
          gl.draw_arrays(GL::TRIANGLE_STRIP, 0, 4);
          // 2) الظلال
          let ns = (self.shad.len() / 3) as i32;
          if ns > 0 {
              gl.disable(GL::DEPTH_TEST);
              gl.enable(GL::BLEND);
              gl.use_program(Some(&self.sp));
              gl.uniform_matrix4fv_with_f32_array(self.s_vp.as_ref(), false, vp);
              gl.bind_vertex_array(Some(&self.s_vao));
              gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.sb));
              gl.buffer_data_with_u8_array(GL::ARRAY_BUFFER, bytes(&self.shad), GL::STREAM_DRAW);
              gl.draw_arrays_instanced(GL::TRIANGLE_STRIP, 0, 4, ns);
              gl.disable(GL::BLEND);
              gl.enable(GL::DEPTH_TEST);
          }
          // 3) كل الشخصيات (instancing)
          let ni = (self.inst.len() / 10) as i32;
          if ni > 0 {
              gl.use_program(Some(&self.bp));
              gl.uniform_matrix4fv_with_f32_array(self.b_vp.as_ref(), false, vp);
              gl.uniform3f(self.b_sky.as_ref(), SKY[0], SKY[1], SKY[2]);
              gl.bind_vertex_array(Some(&self.b_vao));
              gl.bind_buffer(GL::ARRAY_BUFFER, Some(&self.ib));
              gl.buffer_data_with_u8_array(GL::ARRAY_BUFFER, bytes(&self.inst), GL::STREAM_DRAW);
              gl.draw_arrays_instanced(GL::TRIANGLES, 0, 36, ni);
          }
      }
  }

  // ---------------- Math ----------------
  fn persp(fy: f32, a: f32, n: f32, f: f32) -> [f32; 16] {
      let t = 1.0 / (fy / 2.0).tan();
      [t / a, 0., 0., 0., 0., t, 0., 0., 0., 0., (f + n) / (n - f), -1., 0., 0., 2. * f * n / (n - f), 0.]
  }

  fn look_at(e: [f32; 3], t: [f32; 3]) -> [f32; 16] {
      let (mut zx, mut zy, mut zz) = (e[0] - t[0], e[1] - t[1], e[2] - t[2]);
      let l = (zx * zx + zy * zy + zz * zz).sqrt().max(1e-6);
      zx /= l; zy /= l; zz /= l;
      let (mut xx, xy, mut xz) = (zz, 0.0f32, -zx);
      let l = (xx * xx + xz * xz).sqrt().max(1e-6);
      xx /= l; xz /= l;
      let (yx, yy, yz) = (zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx);
      [xx, yx, zx, 0., xy, yy, zy, 0., xz, yz, zz, 0.,
       -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1.]
  }

  fn mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
      let mut o = [0.0; 16];
      for c in 0..4 { for r in 0..4 {
          let mut s = 0.0;
          for k in 0..4 { s += a[k * 4 + r] * b[c * 4 + k]; }
          o[c * 4 + r] = s;
      } }
      o
  }

  fn wrap(a: f32) -> f32 { (a + PI).rem_euclid(TAU) - PI }
  fn dz(v: f32) -> f32 { if v.abs() < DEAD { 0.0 } else { (v - DEAD * v.signum()) / (1.0 - DEAD) } }
  fn u16le(d: &[u8], o: usize) -> u16 { u16::from_le_bytes([d[o], d[o + 1]]) }
  fn f32le(d: &[u8], o: usize) -> f32 { f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) }

  // ---------------- DOM helpers ----------------
  fn el<T: JsCast>(doc: &Document, id: &str) -> Result<T, JsValue> {
      doc.get_element_by_id(id).ok_or_else(|| JsValue::from_str(id))?.dyn_into::<T>().map_err(Into::into)
  }

  fn on(t: &EventTarget, ev: &str, f: impl FnMut(Event) + 'static) {
      let c = Closure::<dyn FnMut(Event)>::new(f);
      let _ = t.add_event_listener_with_callback(ev, c.as_ref().unchecked_ref());
      c.forget();
  }

  fn show(e: &HtmlElement, v: bool) { let _ = e.style().set_property("display", if v { "block" } else { "none" }); }

  fn css_rgb(c: [f32; 3]) -> String {
      format!("rgb({},{},{})", (c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8)
  }

  // ---------------- Labels (أسماء + فقاعات الشات) ----------------
  struct Label { root: HtmlElement, bub: HtmlElement, name: HtmlElement, shown: bool, bubble_until: f64 }

  fn new_label(doc: &Document, parent: &HtmlElement, name: &str, c: [f32; 3]) -> Label {
      let mk = |cls: &str| -> HtmlElement {
          let e: HtmlElement = doc.create_element("div").unwrap().unchecked_into();
          e.set_class_name(cls);
          e
      };
      let (root, bub, nm) = (mk("lbl"), mk("b"), mk("n"));
      let _ = root.append_child(&bub);
      let _ = root.append_child(&nm);
      let _ = parent.append_child(&root);
      let l = Label { root, bub, name: nm, shown: true, bubble_until: 0.0 };
      set_name(&l, name, c);
      l
  }

  fn set_name(l: &Label, name: &str, c: [f32; 3]) {
      l.name.set_text_content(Some(name));
      let _ = l.name.style().set_property("border-color", &css_rgb(c));
  }

  fn place(l: &mut Label, vp: &[f32; 16], p: [f32; 3], w: f32, h: f32, near: bool) {
      let cw = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
      let vis = near && cw > 0.1;
      if vis != l.shown { show(&l.root, vis); l.shown = vis; }
      if !vis { return; }
      let cx = (vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12]) / cw;
      let cy = (vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13]) / cw;
      let (sx, sy) = ((cx * 0.5 + 0.5) * w, (0.5 - cy * 0.5) * h);
      let _ = l.root.style().set_property("transform", &format!("translate3d({sx:.1}px,{sy:.1}px,0) translate(-50%,-100%)"));
  }

  fn expire(l: &mut Label, now: f64) {
      if l.bubble_until > 0.0 && now > l.bubble_until {
          l.bub.set_text_content(None);
          l.bubble_until = 0.0;
      }
  }

  // ---------------- Gamepad ----------------
  #[derive(Default, Clone, Copy)]
  struct Pad { lx: f32, ly: f32, rx: f32, ry: f32, a: bool, run: bool, start: bool, zin: bool, zout: bool }

  fn poll_pad(nav: &Navigator) -> Option<Pad> {
      let pads = nav.get_gamepads().ok()?;
      for v in pads.iter() {
          let Ok(gp) = v.dyn_into::<Gamepad>() else { continue };
          if !gp.connected() { continue; }
          let ax = gp.axes();
          let a = |i: u32| dz(ax.get(i).as_f64().unwrap_or(0.0) as f32);
          let bt = gp.buttons();
          let b = |i: u32| bt.get(i).dyn_into::<GamepadButton>().map(|x| x.pressed() || x.value() > 0.5).unwrap_or(false);
          return Some(Pad {
              lx: a(0), ly: a(1), rx: a(2), ry: a(3),
              a: b(0), run: b(10) || b(7) || b(5), start: b(9), zin: b(12), zout: b(13),
          });
      }
      None
  }

  // ---------------- Game ----------------
  struct Remote { x: f32, y: f32, z: f32, yaw: f32, tx: f32, ty: f32, tz: f32, tyaw: f32, ph: f32, amp: f32, rgb: [f32; 3], label: Label, buf: VecDeque<(f64, [f32; 4])>, off: f64, jit: f64, dly: f64, st: [u8; 4] }

// ---- Jitter buffer: نعرض اللاعب الثاني بخطه الزمني الحقيقي ----
fn rs_push(r: &mut Remote, now: f64, ts: f64, p: [f32; 4]) {
    if let Some(&(lt, lp)) = r.buf.back() {
        if ts <= lt && lt - ts < 60000.0 { return; }
        let (dx, dy, dz) = (p[0] - lp[0], p[1] - lp[1], p[2] - lp[2]);
        if ts <= lt || dx * dx + dy * dy + dz * dz > 100.0 {
            r.buf.clear();
            r.off = f64::NAN;
        } else if ts - lt > 150.0 {
            r.buf.push_back((ts - SEND_MS, lp));
        }
    }
    let o = now - ts;
    if r.off.is_nan() {
        r.off = o;
    } else {
        r.off = (r.off + 0.005).min(o);
        let e = o - r.off;
        r.jit = if e > r.jit { e } else { r.jit + (e - r.jit) * 0.01 };
    }
    r.buf.push_back((ts, p));
    while r.buf.len() > 64 { r.buf.pop_front(); }
}

fn rs_sample(b: &VecDeque<(f64, [f32; 4])>, t: f64) -> Option<[f32; 4]> {
    let (t0, a) = *b.front()?;
    if t <= t0 { return Some(a); }
    for i in 1..b.len() {
        let (t1, q) = b[i];
        if t <= t1 {
            let (t0, a) = b[i - 1];
            let k = ((t - t0) / (t1 - t0).max(1.0)) as f32;
            return Some([a[0] + (q[0] - a[0]) * k, a[1] + (q[1] - a[1]) * k, a[2] + (q[2] - a[2]) * k, a[3] + wrap(q[3] - a[3]) * k]);
        }
    }
    b.back().map(|v| v.1)
}

fn rs_interp(r: &mut Remote, now: f64, dt: f32) {
    let tgt = (r.jit + 12.0).clamp(24.0, 300.0);
    r.dly += (tgt - r.dly) * (dt as f64 * 2.0).min(1.0);
    let t = now - r.off - r.dly;
    while r.buf.len() > 2 && r.buf[1].0 <= t { r.buf.pop_front(); }
    if let Some(p) = rs_sample(&r.buf, t) { r.tx = p[0]; r.ty = p[1].max(0.0); r.tz = p[2]; r.tyaw = p[3]; }
}

  // NET_PING_V1
#[derive(Default)]
struct NetPing { last: f64, seq: u32, waiting: Option<(u32, f64)>, rtt: Option<f64> }
// ELYTRA_V3
fn el_f32(d: &[u8], i: usize) -> f32 { f32::from_le_bytes([d[i], d[i + 1], d[i + 2], d[i + 3]]) }

#[derive(Default)]
struct Elytra { on: bool, held: bool, boost_block: bool, boosting: bool, v: [f32; 3], kv: [f32; 2], roll: f32, fuel: f32 }

#[derive(Default)]
struct Arms {
    weapon: u8, hp: u8, ammo: u8, reload_until: f64, next: f64, swing: f64,
    dead: bool, fire_held: bool, sent_flags: u8,
    fx: Vec<(f64, [f32; 3], [f32; 3])>,
    rswing: HashMap<u16, f64>,
}
struct Game {
    ely: Elytra,
    arms: Arms,
    ping: NetPing,
      win: Window, doc: Document, nav: Navigator, canvas: HtmlCanvasElement, gl: GL, r: Renderer,
      ui: HtmlElement, hud: HtmlElement, labels: HtmlElement, chatlog: HtmlElement, chatin: HtmlInputElement,
      joy: HtmlElement, knob: HtmlElement, name_in: HtmlInputElement, col_in: HtmlInputElement,
      ws: Option<WebSocket>, started: bool, joined: bool, my_id: u16, name: String, rgb: [u8; 3],
      x: f32, y: f32, z: f32, vy: f32, yaw: f32, face: f32, pitch: f32, dist: f32, ph: f32, amp: f32, grounded: bool,
      keys: [bool; 6], touch: bool, chat_open: bool,
      joy_id: Option<i32>, joy_o: (f32, f32), joy_v: (f32, f32), look_id: Option<i32>, look_p: (f32, f32), jump_tap: bool,
      infos: HashMap<u16, (String, [f32; 3])>, remotes: HashMap<u16, Remote>, me_label: Option<Label>,
      chat_lines: VecDeque<(f64, Element)>, seen_buf: Vec<u16>,
      now: f64, last_t: f64, last_send: f64, last_ka: f64, sent: [f32; 4], fps_n: u32, fps_t: f64, pad_on: bool,
  }

  impl Game {
      fn fullscreen(&self) {
          if self.doc.fullscreen_element().is_none() {
              if let Some(e) = self.doc.document_element() { let _ = e.request_fullscreen(); }
          }
      }

      fn lock_landscape(&self) {
          if !self.touch { return; }
          if let Ok(s) = self.win.screen() {
              if let Ok(p) = s.orientation().lock(OrientationLockType::Landscape) {
                  NOOP.with(|c| { let _ = p.catch(c); });
              }
          }
      }

      fn toggle_fs(&self) {
          if self.doc.fullscreen_element().is_some() { self.doc.exit_fullscreen(); } else { self.fullscreen(); }
      }

      fn set_touch(&mut self) {
          if !self.touch {
              self.touch = true;
              if let Some(b) = self.doc.body() { let _ = b.set_attribute("data-touch", "1"); }
          }
      }

      fn send(&self, b: &[u8]) {
          if let Some(ws) = &self.ws {
              if ws.ready_state() == WebSocket::OPEN { let _ = ws.send_with_u8_array(b); }
          }
      }

      fn send_join(&self) {
          let mut b = vec![1u8, self.rgb[0], self.rgb[1], self.rgb[2]];
          b.extend_from_slice(self.name.as_bytes());
          self.send(&b);
      }

      fn open_chat(&mut self) {
          if !self.joined { return; }
          self.chat_open = true;
          self.keys = [false; 6];
          show(&self.chatin, true);
          let _ = self.chatin.focus();
          self.doc.exit_pointer_lock();
      }

      fn close_chat(&mut self) {
          self.chat_open = false;
          self.chatin.set_value("");
          let _ = self.chatin.blur();
          show(&self.chatin, false);
          if !self.touch { self.canvas.request_pointer_lock(); }
      }

      fn send_chat(&self) {
          let t: String = self.chatin.value().trim().chars().take(100).collect();
          if t.is_empty() { return; }
          let mut b = vec![3u8];
          b.extend_from_slice(t.as_bytes());
          self.send(&b);
      }

      fn add_chat(&mut self, name: &str, text: &str) {
          let Ok(e) = self.doc.create_element("div") else { return };
          e.set_text_content(Some(&format!("{name}: {text}")));
          let _ = self.chatlog.append_child(&e);
          self.chat_lines.push_back((self.now, e));
          while self.chatlog.child_element_count() > CHAT_LINES {
              if let Some((_, old)) = self.chat_lines.pop_front() { old.remove(); } else { break; }
          }
      }

      fn key(&mut self, e: &KeyboardEvent, down: bool) {
          if !self.started { return; }
          if self.chat_open {
              if down {
                  match e.key().as_str() {
                      "Enter" => { e.prevent_default(); self.send_chat(); self.close_chat(); }
                      "Escape" => self.close_chat(),
                      _ => {}
                  }
              }
              return;
          }
          let code = e.code();
          let i = match code.as_str() {
              "KeyW" | "ArrowUp" => 0,
              "KeyS" | "ArrowDown" => 1,
              "KeyA" | "ArrowLeft" => 2,
              "KeyD" | "ArrowRight" => 3,
              "ShiftLeft" | "ShiftRight" => 4,
              "Space" => 5,
              "Enter" if down => { e.prevent_default(); self.open_chat(); return; }
              "KeyF" if down => { self.toggle_fs(); return; }
              _ => return,
          };
          self.keys[i] = down;
          if i == 5 || code.starts_with("Arrow") { e.prevent_default(); }
      }

      fn pdown(&mut self, e: &PointerEvent) {
          if !self.started || self.chat_open { return; }
          if e.pointer_type() == "touch" {
              self.set_touch();
              e.prevent_default();
              let (x, y) = (e.client_x() as f32, e.client_y() as f32);
              if x < self.canvas.client_width() as f32 * 0.45 && self.joy_id.is_none() {
                  self.joy_id = Some(e.pointer_id());
                  self.joy_o = (x, y);
                  self.joy_v = (0.0, 0.0);
                  let st = self.joy.style();
                  let _ = st.set_property("left", &format!("{x}px"));
                  let _ = st.set_property("top", &format!("{y}px"));
                  let _ = self.knob.style().set_property("transform", "translate(0,0)");
                  show(&self.joy, true);
              } else if self.look_id.is_none() {
                  self.look_id = Some(e.pointer_id());
                  self.look_p = (x, y);
              }
          } else if self.doc.pointer_lock_element().is_none() {
              self.canvas.request_pointer_lock();
          }
      }

      fn pmove(&mut self, e: &PointerEvent) {
          let id = Some(e.pointer_id());
          let (x, y) = (e.client_x() as f32, e.client_y() as f32);
          if id == self.joy_id && id.is_some() {
              let (mut dx, mut dy) = (x - self.joy_o.0, y - self.joy_o.1);
              let l = dx.hypot(dy);
              if l > JOY_R { dx *= JOY_R / l; dy *= JOY_R / l; }
              self.joy_v = (dx / JOY_R, dy / JOY_R);
              let _ = self.knob.style().set_property("transform", &format!("translate({dx:.0}px,{dy:.0}px)"));
          } else if id == self.look_id && id.is_some() {
              self.yaw -= (x - self.look_p.0) * 0.006;
              self.pitch = (self.pitch + (y - self.look_p.1) * 0.004).clamp(-1.25, 1.3);
              self.look_p = (x, y);
          }
      }

      fn pup(&mut self, e: &PointerEvent) {
          let id = Some(e.pointer_id());
          if id == self.joy_id { self.joy_id = None; self.joy_v = (0.0, 0.0); show(&self.joy, false); }
          if id == self.look_id { self.look_id = None; }
      }

      fn ping_clock(&self) -> f64 {
    self.win.performance().map(|p| p.now()).unwrap_or(self.now)
}
fn ping_tick(&mut self) {
    if !self.joined { return; }
    let now = self.ping_clock();
    if let Some((_, sent)) = self.ping.waiting {
        if now - sent < 3000.0 { return; }
        self.ping.waiting = None;
        self.ping.rtt = None;
    }
    if now - self.ping.last < 1000.0 { return; }
    if !self.ws.as_ref().map(|w| w.ready_state() == WebSocket::OPEN).unwrap_or(false) { return; }
    self.ping.seq = self.ping.seq.wrapping_add(1);
    let mut b = [0u8; 5];
    b[0] = 13;
    b[1..5].copy_from_slice(&self.ping.seq.to_le_bytes());
    self.ping.last = now;
    self.ping.waiting = Some((self.ping.seq, now));
    self.send(&b);
}
fn ping_pong(&mut self, d: &[u8]) {
    if d.len() != 5 { return; }
    let seq = u32::from_le_bytes([d[1], d[2], d[3], d[4]]);
    if let Some((expected, sent)) = self.ping.waiting {
        if seq == expected {
            let elapsed = self.ping_clock() - sent;
            if elapsed.is_finite() && elapsed >= 0.0 { self.ping.rtt = Some(elapsed); }
            self.ping.waiting = None;
        }
    }
}
fn on_net(&mut self, d: &[u8]) {
          let Some(&t) = d.first() else { return };
          match t {
            8 if d.len() >= 16 => { self.on_hit(d); }
            9 if d.len() >= 18 => { self.on_fx(d); }
            10 if d.len() >= 9 => { self.on_respawn(d); }
            11 if d.len() >= 6 => { self.on_kill(d); }
    13 if d.len() == 5 => { self.ping_pong(d); }
              1 if d.len() >= 11 => {
                  self.my_id = u16le(d, 1);
                  self.x = f32le(d, 3);
                  self.z = f32le(d, 7);
                  self.joined = true;
                  if self.name.is_empty() { self.name = format!("Player{}", self.my_id); }
                  if self.me_label.is_none() {
                      let c = self.rgb.map(|v| v as f32 / 255.0);
                      self.me_label = Some(new_label(&self.doc, &self.labels, &self.name, c));
                  }
              }
              2 if d.len() >= 3 => {
                  let n = u16le(d, 1) as usize;
                  if d.len() < 3 + n * 18 { return; }
                  self.seen_buf.clear();
                  for i in 0..n {
                      let o = 3 + i * 18;
                      let id = u16le(d, o);
                      if id == self.my_id { continue; }
                      let (x, y, z, yw) = (f32le(d, o + 2), f32le(d, o + 6), f32le(d, o + 10), f32le(d, o + 14));
                      self.seen_buf.push(id);
                      let (doc, labels, infos) = (&self.doc, &self.labels, &self.infos);
                      let r = self.remotes.entry(id).or_insert_with(|| {
                          let (nm, c) = infos.get(&id).cloned().unwrap_or_else(|| ("...".into(), [1.0; 3]));
                          Remote { x, y, z, yaw: yw, tx: x, ty: y, tz: z, tyaw: yw, ph: 0.0, amp: 0.0, rgb: c, label: new_label(doc, labels, &nm, c), buf: VecDeque::new(), off: f64::NAN, jit: 40.0, dly: 60.0, st: [0; 4] }
                      });
                      r.tx = x; r.ty = y; r.tz = z; r.tyaw = yw;
                  }
                  let seen = &self.seen_buf;
                  self.remotes.retain(|id, r| {
                      let keep = seen.contains(id);
                      if !keep { r.label.root.remove(); }
                      keep
                  });
              }
              14 if d.len() >= 3 => {
                  let n = u16le(d, 1) as usize;
                  if d.len() < 3 + n * 26 { return; }
                  let now = self.now;
                  self.seen_buf.clear();
                  for i in 0..n {
                      let o = 3 + i * 26;
                      let id = u16le(d, o);
                      if id == self.my_id { continue; }
                      let p = [f32le(d, o + 2), f32le(d, o + 6), f32le(d, o + 10), f32le(d, o + 14)];
                      let ts = u32::from_le_bytes([d[o + 18], d[o + 19], d[o + 20], d[o + 21]]) as f64;
                      let st = [d[o + 22], d[o + 23], d[o + 24], d[o + 25]];
                      self.seen_buf.push(id);
                      let (doc, labels, infos) = (&self.doc, &self.labels, &self.infos);
                      let r = self.remotes.entry(id).or_insert_with(|| {
                          let (nm, c) = infos.get(&id).cloned().unwrap_or_else(|| ("...".into(), [1.0; 3]));
                          Remote { x: p[0], y: p[1], z: p[2], yaw: p[3], tx: p[0], ty: p[1], tz: p[2], tyaw: p[3], ph: 0.0, amp: 0.0, rgb: c,
                              label: new_label(doc, labels, &nm, c), buf: VecDeque::new(), off: f64::NAN, jit: 40.0, dly: 60.0, st }
                      });
                      r.st = st;
                      rs_push(r, now, ts, p);
                  }
                  let seen = &self.seen_buf;
                  self.remotes.retain(|id, r| {
                      let keep = seen.contains(id);
                      if !keep { r.label.root.remove(); }
                      keep
                  });
              }
              3 => {
                  let mut o = 1;
                  while o + 6 <= d.len() {
                      let id = u16le(d, o);
                      let c = [d[o + 2], d[o + 3], d[o + 4]].map(|v| v as f32 / 255.0);
                      let l = d[o + 5] as usize;
                      o += 6;
                      if o + l > d.len() { break; }
                      let nm = String::from_utf8_lossy(&d[o..o + l]).into_owned();
                      o += l;
                      if let Some(r) = self.remotes.get_mut(&id) { r.rgb = c; set_name(&r.label, &nm, c); }
                      self.infos.insert(id, (nm, c));
                  }
              }
              4 if d.len() >= 3 => {
                  let id = u16le(d, 1);
                  self.infos.remove(&id);
                  if let Some(r) = self.remotes.remove(&id) { r.label.root.remove(); }
              }
              5 if d.len() >= 3 => {
                  let id = u16le(d, 1);
                  let text = String::from_utf8_lossy(&d[3..]).into_owned();
                  let nm = if id == self.my_id { self.name.clone() } else { self.infos.get(&id).map(|i| i.0.clone()).unwrap_or_default() };
                  self.add_chat(&nm, &text);
                  let until = self.now + BUBBLE_MS;
                  let lbl = if id == self.my_id { self.me_label.as_mut() } else { self.remotes.get_mut(&id).map(|r| &mut r.label) };
                  if let Some(l) = lbl { l.bub.set_text_content(Some(&text)); l.bubble_until = until; }
              }
              _ => {}
          }
      }

      fn frame(&mut self, t: f64) -> bool {
          let dt = (((t - self.last_t) / 1000.0) as f32).clamp(0.0, 0.1);
          self.last_t = t;
          self.now = t;
          let pad = poll_pad(&self.nav);
          self.pad_on = pad.is_some();
          let pad = pad.unwrap_or_default();
          let want_join = !self.started && (pad.start || pad.a);
          if self.joined { self.control(dt, pad); }
          self.animate(dt);
          self.net();
    self.ping_tick();
          self.combat_tick();
          self.render();
          self.hud_update(t);
          want_join
      }

          fn look_dir(&self) -> [f32; 3] {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        [-sy * cp, -sp, -cy * cp]
    }

    fn net_flags(&self) -> u8 {
        (if self.ely.on { 1u8 } else { 0 }) | ((self.arms.weapon & 3) << 1) | (if self.ely.boosting { 0x10u8 } else { 0 })
    }

    fn ely_open(&mut self) {
        if self.arms.hp == 0 || self.ely.on { return; }
        let (sy, cy) = self.yaw.sin_cos();
        if self.grounded { self.y = self.y.max(0.3); self.vy = self.vy.max(9.0); }
        self.grounded = false;
        self.ely.on = true;
        self.ely.roll = 0.0;
        self.ely.v = [-sy * 0.35 + self.ely.kv[0] / 20.0, self.vy / 20.0, -cy * 0.35 + self.ely.kv[1] / 20.0];
        self.ely.kv = [0.0; 2];
    }

    fn ely_close(&mut self) {
        if !self.ely.on { return; }
        self.ely.on = false;
        self.ely.boosting = false;
        self.vy = self.ely.v[1] * 20.0;
        self.ely.kv = [self.ely.v[0] * 20.0, self.ely.v[2] * 20.0];
    }

    fn ely_toggle(&mut self) {
        if !self.joined || self.chat_open { return; }
        if self.ely.on { self.ely_close(); } else { self.ely_open(); self.ely.boost_block = true; }
    }

    fn elytra_step(&mut self, dt: f32, boost: bool, close: bool) {
        if close { self.ely_close(); return; }
        let t = (dt * 20.0).min(3.0);
        let look = self.look_dir();
        let hl = (look[0] * look[0] + look[2] * look[2]).sqrt();
        let pitch = self.pitch;
        let mut v = self.ely.v;
        let hv = (v[0] * v[0] + v[2] * v[2]).sqrt();
        let lift = hl * hl;
        v[1] += (-0.08 + lift * 0.06) * t;
        if v[1] < 0.0 && hl > 0.001 {
            let a = v[1] * -0.1 * lift * t;
            v[1] += a;
            v[0] += look[0] * a / hl;
            v[2] += look[2] * a / hl;
        }
        if pitch < 0.0 && hl > 0.001 {
            let a = hv * (-pitch).sin() * 0.04 * t;
            v[1] += a * 3.2;
            v[0] -= look[0] * a / hl;
            v[2] -= look[2] * a / hl;
        }
        if hl > 0.001 {
            let f = 1.0 - 0.9f32.powf(t);
            v[0] += (look[0] / hl * hv - v[0]) * f;
            v[2] += (look[2] / hl * hv - v[2]) * f;
        }
        self.ely.boosting = boost && self.ely.fuel > 0.02;
        if self.ely.boosting {
            let f = 1.0 - 0.5f32.powf(t);
            for i in 0..3 { v[i] += look[i] * 0.1 * t + (look[i] * 1.5 - v[i]) * f; }
            self.ely.fuel = (self.ely.fuel - dt * 0.3).max(0.0);
        } else {
            self.ely.fuel = (self.ely.fuel + dt * 0.06).min(1.0);
        }
        v[0] *= 0.99f32.powf(t);
        v[1] *= 0.98f32.powf(t);
        v[2] *= 0.99f32.powf(t);
        let s = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if s > 2.75 { for c in v.iter_mut() { *c *= 2.75 / s; } }
        self.x += v[0] * t;
        self.y += v[1] * t;
        self.z += v[2] * t;
        let hv2 = (v[0] * v[0] + v[2] * v[2]).sqrt();
        if hv2 > 0.02 {
            let nf = (-v[0]).atan2(-v[2]);
            let turn = wrap(nf - self.face) / dt.max(0.001);
            self.face = nf;
            let tr = (turn * 0.35).clamp(-0.9, 0.9);
            self.ely.roll += (tr - self.ely.roll) * (dt * 6.0).min(1.0);
        }
        self.ely.v = v;
        self.vy = v[1] * 20.0;
        self.grounded = false;
        if self.y >= 400.0 { self.y = 400.0; if self.ely.v[1] > 0.0 { self.ely.v[1] = 0.0; } }
        if self.y <= 0.0 {
            self.y = 0.0;
            self.vy = 0.0;
            self.grounded = true;
            self.ely.on = false;
            self.ely.boosting = false;
            self.ely.kv = [v[0] * 5.0, v[2] * 5.0];
        }
    }

    fn knock_step(&mut self, dt: f32) {
        if self.ely.kv[0] == 0.0 && self.ely.kv[1] == 0.0 { return; }
        self.x += self.ely.kv[0] * dt;
        self.z += self.ely.kv[1] * dt;
        let d = (-dt * if self.grounded { 8.0 } else { 1.5 }).exp();
        self.ely.kv[0] *= d;
        self.ely.kv[1] *= d;
        if self.ely.kv[0].abs() + self.ely.kv[1].abs() < 0.05 { self.ely.kv = [0.0; 2]; }
    }

    fn set_weapon(&mut self, w: u8) {
        if self.arms.hp == 0 { return; }
        self.arms.weapon = if w > 2 { 0 } else { w };
        self.arms.fire_held = false;
    }

    fn reload(&mut self) {
        if self.arms.weapon != 2 || self.arms.ammo >= 12 || self.arms.reload_until > 0.0 || self.arms.hp == 0 { return; }
        self.arms.reload_until = self.now + 2250.0;
        self.send(&[7]);
    }

    fn combat_tick(&mut self) {
        if self.arms.reload_until > 0.0 && self.now >= self.arms.reload_until {
            self.arms.ammo = 12;
            self.arms.reload_until = 0.0;
        }
        if self.arms.fire_held { self.try_attack(); }
        let now = self.now;
        self.arms.fx.retain(|f| now - f.0 < 140.0);
        self.arms.rswing.retain(|_, t| now - *t < 350.0);
    }

    fn try_attack(&mut self) {
        if !self.joined || self.chat_open || self.arms.hp == 0 { return; }
        let w = self.arms.weapon;
        if w == 0 || self.now < self.arms.next { return; }
        if w == 2 {
            if self.arms.reload_until > 0.0 { return; }
            if self.arms.ammo == 0 { self.reload(); return; }
        }
        self.arms.next = self.now + if w == 1 { 580.0 } else { 150.0 };
        let dir = self.look_dir();
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let cam = [sy * cp * self.dist, 1.4 + self.y + sp * self.dist, cy * cp * self.dist];
        let eye = [0.0f32, self.y + 1.5, 0.0];
        let mut best: Option<(f32, u16, [f32; 3], bool)> = None;
        for (&id, r) in self.remotes.iter() {
            if r.st[0] & 0x40 != 0 { continue; }
            let base = [r.x - self.x, r.y, r.z - self.z];
            if w == 1 {
                let p = [base[0], base[1] + 1.0, base[2]];
                let v = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
                let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if d > 3.4 { continue; }
                let hd = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt().max(0.001);
                let hvl = (v[0] * v[0] + v[2] * v[2]).sqrt().max(0.001);
                let facing = (v[0] * dir[0] + v[2] * dir[2]) / hd / hvl;
                if d > 1.2 && facing < 0.35 { continue; }
                if best.map_or(true, |b| d < b.0) { best = Some((d, id, p, false)); }
            } else {
                for &(h, py, tol) in [(true, 1.65f32, 0.32f32), (false, 1.05f32, 0.62f32)].iter() {
                    let p = [base[0], base[1] + py, base[2]];
                    let v = [p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]];
                    let along = v[0] * dir[0] + v[1] * dir[1] + v[2] * dir[2];
                    if along <= 0.0 { continue; }
                    let q = [v[0] - dir[0] * along, v[1] - dir[1] * along, v[2] - dir[2] * along];
                    let perp = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
                    let e = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
                    let fe = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
                    if perp > tol + along * 0.004 || fe > 150.0 { continue; }
                    if best.map_or(true, |b| along < b.0) { best = Some((along, id, p, h)); }
                    break;
                }
            }
        }
        let (target, aim, head) = match best {
            Some((_, id, p, h)) => (id, [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]], h),
            None => (0u16, dir, false),
        };
        let mut b = [0u8; 16];
        b[0] = 6;
        b[1] = w | (if head { 0x80u8 } else { 0 });
        b[2..4].copy_from_slice(&target.to_le_bytes());
        for i in 0..3 { b[4 + i * 4..8 + i * 4].copy_from_slice(&aim[i].to_le_bytes()); }
        self.send(&b);
        self.arms.swing = self.now;
        if !self.ely.on { self.face = (-dir[0]).atan2(-dir[2]); }
        if w == 2 {
            self.arms.ammo -= 1;
            let l = (aim[0] * aim[0] + aim[1] * aim[1] + aim[2] * aim[2]).sqrt().max(0.001);
            let reach = if target != 0 { l } else { 150.0 };
            let (fx, fz) = (-self.face.sin(), -self.face.cos());
            let (rx, rz) = (self.face.cos(), -self.face.sin());
            let s = [self.x + rx * 0.42 + fx * 0.6, self.y + if self.ely.on { 0.52 } else { 0.82 }, self.z + rz * 0.42 + fz * 0.6];
            let e = [self.x + aim[0] / l * reach, eye[1] + aim[1] / l * reach, self.z + aim[2] / l * reach];
            self.arms.fx.push((self.now, s, e));
            if self.arms.ammo == 0 { self.reload(); }
        }
    }

    fn on_hit(&mut self, d: &[u8]) {
        let (ix, iy, iz) = (el_f32(d, 3), el_f32(d, 7), el_f32(d, 11));
        if ix.is_finite() && iy.is_finite() && iz.is_finite() {
            if self.ely.on {
                self.ely.v[0] += ix / 20.0;
                self.ely.v[1] += iy / 20.0;
                self.ely.v[2] += iz / 20.0;
            } else {
                self.ely.kv[0] += ix;
                self.ely.kv[1] += iz;
                if iy > 0.0 { self.vy = self.vy.max(0.0) + iy; self.grounded = false; }
            }
        }
        self.arms.hp = d[15];
        if self.arms.hp == 0 {
            self.arms.dead = true;
            self.arms.fire_held = false;
            self.ely_close();
        }
    }

    fn on_fx(&mut self, d: &[u8]) {
        let id = u16::from_le_bytes([d[1], d[2]]);
        if id == self.my_id { return; }
        let w = d[3] & 0x7F;
        let e = [el_f32(d, 6), el_f32(d, 10), el_f32(d, 14)];
        if !(e[0].is_finite() && e[1].is_finite() && e[2].is_finite()) { return; }
        let s = match self.remotes.get(&id) {
            Some(r) => [r.x, r.y + if r.st[0] & 1 != 0 { 0.55 } else { 0.85 }, r.z],
            None => return,
        };
        if w == 2 { self.arms.fx.push((self.now, s, e)); }
        self.arms.rswing.insert(id, self.now);
    }

    fn on_respawn(&mut self, d: &[u8]) {
        let (x, z) = (el_f32(d, 1), el_f32(d, 5));
        if x.is_finite() && z.is_finite() { self.x = x; self.z = z; }
        self.y = 0.0;
        self.vy = 0.0;
        self.grounded = true;
        self.ely.on = false;
        self.ely.boosting = false;
        self.ely.kv = [0.0; 2];
        self.ely.v = [0.0; 3];
        self.ely.fuel = 1.0;
        self.arms.hp = 100;
        self.arms.ammo = 12;
        self.arms.reload_until = 0.0;
        self.arms.dead = false;
        self.sent = [f32::NAN; 4];
    }

    fn on_kill(&mut self, d: &[u8]) {
        let killer = u16::from_le_bytes([d[1], d[2]]);
        let victim = u16::from_le_bytes([d[3], d[4]]);
        let w = if d[5] & 0x7F == 1 { "🏏" } else { "🔫" };
        let nm = |g: &Self, id: u16| -> String {
            if id == g.my_id { g.name.clone() } else { g.infos.get(&id).map(|i| i.0.clone()).unwrap_or_else(|| format!("#{id}")) }
        };
        let text = format!("{} {} {}", nm(self, killer), w, nm(self, victim));
        self.add_chat("☠", &text);
    }
    fn control(&mut self, dt: f32, pad: Pad) {
        // ELYTRA_V3 hook
        if self.arms.dead {
            self.jump_tap = false;
            self.ely.on = false;
            self.vy -= 20.0 * dt;
            self.y = (self.y + self.vy * dt).max(0.0);
            if self.y <= 0.0 { self.vy = 0.0; self.grounded = true; }
            return;
        }
        let jd = !self.chat_open && (self.keys[5] || pad.a || self.jump_tap);
        let jp = jd && !self.ely.held;
        self.ely.held = jd;
        if !jd { self.ely.boost_block = false; }
        if self.ely.on {
            self.jump_tap = false;
            let close = !self.chat_open && self.keys[4];
            let boost = jd && !self.ely.boost_block;
            self.elytra_step(dt, boost, close);
            return;
        }
        if jp && !self.grounded && self.y > 0.6 {
            self.ely_open();
            self.ely.boost_block = true;
            self.jump_tap = false;
            return;
        }
        self.ely.fuel = (self.ely.fuel + dt * if self.grounded { 0.5 } else { 0.06 }).min(1.0);
        self.knock_step(dt);

          let (k, pad) = if self.chat_open { ([false; 6], Pad::default()) } else { (self.keys, pad) };
          let mut ix = (k[3] as i32 - k[2] as i32) as f32 + pad.lx;
          let mut iy = (k[0] as i32 - k[1] as i32) as f32 - pad.ly;
          if self.joy_id.is_some() { ix += self.joy_v.0; iy -= self.joy_v.1; }
          let mag = ix.hypot(iy);
          if mag > 1.0 { ix /= mag; iy /= mag; }
          let mag = mag.min(1.0);
          let run = k[4] || pad.run || (self.joy_id.is_some() && mag > 0.95);

          self.yaw -= pad.rx * 2.6 * dt;
          self.pitch = (self.pitch + pad.ry * 1.8 * dt).clamp(-1.25, 1.3);
          if pad.zin { self.dist *= 1.0 - dt * 1.5; }
          if pad.zout { self.dist *= 1.0 + dt * 1.5; }
          self.dist = self.dist.clamp(3.0, 25.0);

          let (s, c) = self.yaw.sin_cos();
          let (mx, mz) = (-s * iy + c * ix, -c * iy - s * ix);
          let sp = if run { RUN } else { WALK };
          let moving = mag > 0.01;
          if moving {
              self.x += mx * sp * dt;
              self.z += mz * sp * dt;
              self.ph += sp * mag * dt * 1.8;
              self.face += wrap((-mx).atan2(-mz) - self.face) * (dt * 12.0).min(1.0);
          }
          self.amp += ((if moving { mag.max(0.4) } else { 0.0 }) - self.amp) * (dt * 10.0).min(1.0);

          if (k[5] || pad.a || self.jump_tap) && self.grounded { self.vy = JUMP; self.grounded = false; }
          self.jump_tap = false;
          self.vy -= GRAVITY * dt;
          self.y += self.vy * dt;
          if self.y <= 0.0 { self.y = 0.0; self.vy = 0.0; self.grounded = true; }
      }

      fn animate(&mut self, dt: f32) {
          let k = 1.0 - (-dt * 25.0).exp();
          let now = self.now;
          for r in self.remotes.values_mut() {
              let k = if r.off.is_nan() { k } else { rs_interp(r, now, dt); 1.0 };
              let (dx, dz) = ((r.tx - r.x) * k, (r.tz - r.z) * k);
              r.x += dx; r.z += dz;
              r.y += (r.ty - r.y) * k;
              r.yaw += wrap(r.tyaw - r.yaw) * k;
              let m = dx.hypot(dz);
              r.ph += m * 1.8;
              let target = if dt > 0.0 { (m / dt / 3.0).min(1.0) } else { 0.0 };
              r.amp += (target - r.amp) * (dt * 8.0).min(1.0);
              expire(&mut r.label, now);
          }
          if let Some(l) = self.me_label.as_mut() { expire(l, now); }
          while let Some((t0, _)) = self.chat_lines.front() {
              if now - *t0 < CHAT_LINE_MS { break; }
              if let Some((_, e)) = self.chat_lines.pop_front() { e.remove(); }
          }
      }

      fn net(&mut self) {
          if !self.joined || self.now - self.last_send < SEND_MS { return; }
          self.last_send = self.now;
          let cur = [self.x, self.y, self.z, self.face];
          let fl = self.net_flags();
            if cur != self.sent || fl != self.arms.sent_flags {
              let mut b = [0u8; 24];
              b[0] = 2;
                b[17] = fl;
                let hv = (self.ely.v[0] * self.ely.v[0] + self.ely.v[2] * self.ely.v[2]).sqrt();
                let bp: f32 = if self.ely.on { self.ely.v[1].atan2(hv.max(0.0001)) } else { 0.0 };
                let br: f32 = if self.ely.on { self.ely.roll } else { 0.0 };
                b[18] = ((bp * 60.0).clamp(-127.0, 127.0) as i8) as u8;
                b[19] = ((br * 60.0).clamp(-127.0, 127.0) as i8) as u8;
                self.arms.sent_flags = fl;
              for (i, v) in cur.iter().enumerate() { b[1 + i * 4..5 + i * 4].copy_from_slice(&v.to_le_bytes()); }
              b[20..24].copy_from_slice(&(self.now as u32).to_le_bytes());
              self.send(&b);
              self.sent = cur;
              self.last_ka = self.now;
          } else if self.now - self.last_ka > KEEPALIVE_MS {
              self.send(&[0]);
              self.last_ka = self.now;
          }
      }

      fn render(&mut self) {
          let dpr = self.win.device_pixel_ratio().min(2.0);
          let (cw, ch) = (self.canvas.client_width().max(1), self.canvas.client_height().max(1));
          let (w, h) = ((cw as f64 * dpr) as u32, (ch as f64 * dpr) as u32);
          if self.canvas.width() != w || self.canvas.height() != h {
              self.canvas.set_width(w);
              self.canvas.set_height(h);
              self.gl.viewport(0, 0, w as i32, h as i32);
          }
          let p = persp(1.05, w as f32 / h.max(1) as f32, 0.2, 3000.0);
          let (sp, cp) = self.pitch.sin_cos();
          let ty = 1.4 + self.y;
          let eye = [self.yaw.sin() * cp * self.dist, (ty + sp * self.dist).max(0.3), self.yaw.cos() * cp * self.dist];
          let vp = mul(&p, &look_at(eye, [0.0, ty, 0.0]));

          // Floating Origin: كل شي ينرسم نسبةً للاعب
          self.r.begin();
          if self.joined {
              let c = self.rgb.map(|v| v as f32 / 255.0);
              let lsw = ((self.now - self.arms.swing) / 300.0) as f32;
            if self.ely.on {
                self.r.elytra(0.0, self.y, 0.0, self.face, c, (self.now * 0.001) as f32, self.ely.roll, self.ely.boosting);
            } else {
                self.r.hero(0.0, self.y, 0.0, self.face, c, self.ph, self.amp);
                self.r.folded(0.0, self.y, 0.0, self.face, self.ph, self.amp);
            }
            if self.arms.hp > 0 { self.r.weapon(0.0, self.y, 0.0, self.face, self.arms.weapon, self.ely.on, lsw); }

          }
          for r in self.remotes.values() {
              let (x, z) = (r.x - self.x, r.z - self.z);
              if x.abs() > 1500.0 || z.abs() > 1500.0 { continue; }
              if r.st[0] & 1 != 0 {
                self.r.elytra(x, r.y, z, r.yaw, r.rgb, (self.now * 0.001) as f32, (r.st[2] as i8) as f32 / 60.0, r.st[0] & 0x10 != 0);
            } else {
                self.r.hero(x, r.y, z, r.yaw, r.rgb, r.ph, r.amp);
                self.r.folded(x, r.y, z, r.yaw, r.ph, r.amp);
            }
          }
                  {
            let tnow = self.now;
            for (id, r) in self.remotes.iter() {
                let (x, z) = (r.x - self.x, r.z - self.z);
                if x.abs() > 300.0 || z.abs() > 300.0 { continue; }
                let fl = r.st[0];
                if fl & 0x40 != 0 { continue; }
                let fly = fl & 1 != 0;
                let sw = self.arms.rswing.get(id).map(|t0| ((tnow - *t0) / 300.0) as f32).unwrap_or(9.0);
                self.r.weapon(x, r.y, z, r.yaw, (fl >> 1) & 3, fly, sw);
                let hp = r.st[3];
                if hp > 0 && x * x + z * z < 900.0 {
                    self.r.bar(x, r.y + if fly { 1.3 } else { 2.05 }, z, self.yaw, hp);
                }
            }
            for f in self.arms.fx.iter() {
                let fade = 1.0 - ((tnow - f.0) / 140.0) as f32;
                if fade <= 0.0 { continue; }
                self.r.tracer([f.1[0] - self.x, f.1[1], f.1[2] - self.z], [f.2[0] - self.x, f.2[1], f.2[2] - self.z], fade);
            }
        }
          self.r.draw(&self.gl, &vp, (self.x.rem_euclid(10.0), self.z.rem_euclid(10.0)));

          let (lw, lh) = (cw as f32, ch as f32);
          if let Some(l) = self.me_label.as_mut() { place(l, &vp, [0.0, self.y + 2.25, 0.0], lw, lh, true); }
          for r in self.remotes.values_mut() {
              let (x, z) = (r.x - self.x, r.z - self.z);
              place(&mut r.label, &vp, [x, r.y + 2.25, z], lw, lh, x * x + z * z < LABEL_DIST * LABEL_DIST);
          }
      }

      fn hud_update(&mut self, t: f64) {
          self.fps_n += 1;
          if t - self.fps_t < 500.0 { return; }
          let fps = (self.fps_n as f64 * 1000.0 / (t - self.fps_t)) as u32;
          self.fps_n = 0;
          self.fps_t = t;
          let s = if self.joined {
              format!("FPS {fps} | players {} | X {:.0} Z {:.0}{}", self.remotes.len() + 1, self.x, self.z, if self.pad_on { " | 🎮" } else { "" })
          } else if self.started { "Connecting...".into() } else { format!("FPS {fps}{}", if self.pad_on { " | 🎮 Start" } else { "" }) };
          let s = if self.joined {
    let ping = self.ping.rtt.map(|v| format!("{v:.0} ms")).unwrap_or_else(|| "--".into());
    format!("{s} | Ping {ping}")
} else { s };
        let s = if self.joined {
            let wn = match self.arms.weapon {
                1 => "🏏 Bat".to_string(),
                2 => if self.arms.reload_until > 0.0 { "🔫 Reloading".to_string() } else { format!("🔫 {}/12", self.arms.ammo) },
                _ => "✋ Hands".to_string(),
            };
            let fl = if self.ely.on {
                let v = self.ely.v;
                format!(" | 🪽 {:.0} m/s | ALT {:.0} m | 🚀 {:.0}%", (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() * 20.0, self.y, self.ely.fuel * 100.0)
            } else { String::new() };
            let dead = if self.arms.hp == 0 { " | ☠ Respawning..." } else { "" };
            format!("{s} | ❤ {} | {wn}{fl}{dead}", self.arms.hp)
        } else { s };
          self.hud.set_text_content(Some(&s));
      }
  }

  // ---------------- Join / Network ----------------
  fn join(g: &Rc<RefCell<Game>>, gesture: bool) {
      {
          let mut gm = g.borrow_mut();
          if gm.started { return; }
          gm.started = true;
          gm.name = gm.name_in.value().trim().chars().filter(|c| !c.is_control()).take(16).collect();
          let col = gm.col_in.value();
          let v = u32::from_str_radix(col.trim_start_matches('#'), 16).unwrap_or(0xffffff);
          gm.rgb = [(v >> 16) as u8, (v >> 8) as u8, v as u8];
          if let Ok(Some(s)) = gm.win.local_storage() {
              let _ = s.set_item("name", &gm.name);
              let _ = s.set_item("col", &col);
          }
          show(&gm.ui, false);
          if gesture {
              gm.fullscreen();
              if !gm.touch { gm.canvas.request_pointer_lock(); }
          }
      }
      connect(g);
  }

  fn connect(g: &Rc<RefCell<Game>>) {
      let url = {
          let gm = g.borrow();
          let loc = gm.win.location();
          let proto = if loc.protocol().unwrap_or_default() == "https:" { "wss://" } else { "ws://" };
          format!("{proto}{}/ws", loc.host().unwrap_or_default())
      };
      let Ok(ws) = WebSocket::new(&url) else { return };
      ws.set_binary_type(BinaryType::Arraybuffer);

      let g1 = g.clone();
      let onopen = Closure::<dyn FnMut(Event)>::new(move |_| g1.borrow().send_join());
      ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
      onopen.forget();

      let g2 = g.clone();
      let onmsg = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
          if let Ok(ab) = e.data().dyn_into::<js_sys::ArrayBuffer>() {
              let v = js_sys::Uint8Array::new(&ab).to_vec();
              g2.borrow_mut().on_net(&v);
          }
      });
      ws.set_onmessage(Some(onmsg.as_ref().unchecked_ref()));
      onmsg.forget();

      let g3 = g.clone();
      let onclose = Closure::<dyn FnMut(Event)>::new(move |_| {
          let mut gm = g3.borrow_mut();
          gm.joined = false;
          gm.hud.set_text_content(Some("Disconnected... reconnecting"));
          let loc = gm.win.location();
          let cb = Closure::once_into_js(move || { let _ = loc.reload(); });
          let _ = gm.win.set_timeout_with_callback_and_timeout_and_arguments_0(cb.unchecked_ref(), 2000);
      });
      ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
      onclose.forget();

      g.borrow_mut().ws = Some(ws);
  }

  // ---------------- Input binding ----------------
  fn bind(g: &Rc<RefCell<Game>>) -> Result<(), JsValue> {
      let (win, doc, canvas) = { let gm = g.borrow(); (gm.win.clone(), gm.doc.clone(), gm.canvas.clone()) };
      let go: HtmlElement = el(&doc, "go")?;
      let jump: HtmlElement = el(&doc, "jump")?;
      let chatbtn: HtmlElement = el(&doc, "chatbtn")?;

      { let g = g.clone(); on(&go, "click", move |_| join(&g, true)); }
      { let g = g.clone(); on(&win, "keydown", move |e| g.borrow_mut().key(e.unchecked_ref(), true)); }
      { let g = g.clone(); on(&win, "keyup", move |e| g.borrow_mut().key(e.unchecked_ref(), false)); }
      { let g = g.clone(); on(&win, "blur", move |_| g.borrow_mut().keys = [false; 6]); }
      {
          let g = g.clone();
          on(&win, "mousemove", move |e| {
              let e: &MouseEvent = e.unchecked_ref();
              let mut gm = g.borrow_mut();
              if gm.doc.pointer_lock_element().is_some() {
                  gm.yaw -= e.movement_x() as f32 * 0.0025;
                  gm.pitch = (gm.pitch + e.movement_y() as f32 * 0.0025).clamp(-1.25, 1.3);
              }
          });
      }
      {
          let g = g.clone();
          on(&win, "wheel", move |e| {
              let e: &WheelEvent = e.unchecked_ref();
              let mut gm = g.borrow_mut();
              gm.dist = (gm.dist * if e.delta_y() > 0.0 { 1.1 } else { 0.9 }).clamp(3.0, 25.0);
          });
      }
      { let g = g.clone(); on(&canvas, "pointerdown", move |e| g.borrow_mut().pdown(e.unchecked_ref())); }
      { let g = g.clone(); on(&win, "pointermove", move |e| g.borrow_mut().pmove(e.unchecked_ref())); }
      for ev in ["pointerup", "pointercancel"] {
          let g = g.clone();
          on(&win, ev, move |e| g.borrow_mut().pup(e.unchecked_ref()));
      }
      on(&canvas, "contextmenu", |e| e.prevent_default());
      // أي لمسة أو ضغطة ترجع ملء الشاشة إذا طلعت منه
      { let g = g.clone(); on(&win, "pointerdown", move |_| { let gm = g.borrow(); if gm.joined { gm.fullscreen(); } }); }
      { let g = g.clone(); on(&doc, "fullscreenchange", move |_| { let gm = g.borrow(); if gm.doc.fullscreen_element().is_some() { gm.lock_landscape(); } }); }
      {
          let g = g.clone();
          on(&jump, "pointerdown", move |e| { e.prevent_default(); e.stop_propagation(); g.borrow_mut().jump_tap = true; });
      }
      { let g = g.clone(); on(&chatbtn, "click", move |_| g.borrow_mut().open_chat()); }
    {
        let g = g.clone();
        on(&win, "keydown", move |e| {
            let e: &KeyboardEvent = e.unchecked_ref();
            let mut gm = g.borrow_mut();
            if !gm.joined || gm.chat_open || e.repeat() { return; }
            match e.code().as_str() {
                "KeyG" => gm.ely_toggle(),
                "KeyQ" => { let w = (gm.arms.weapon + 1) % 3; gm.set_weapon(w); }
                "Digit1" => gm.set_weapon(1),
                "Digit2" => gm.set_weapon(2),
                "Digit3" => gm.set_weapon(0),
                "KeyR" => gm.reload(),
                _ => {}
            }
        });
    }
    {
        let g = g.clone();
        on(&win, "mousedown", move |e| {
            let e: &MouseEvent = e.unchecked_ref();
            let mut gm = g.borrow_mut();
            if e.button() == 0 && gm.doc.pointer_lock_element().is_some() {
                gm.arms.fire_held = true;
                gm.try_attack();
            }
        });
    }
    {
        let g = g.clone();
        on(&win, "mouseup", move |e| {
            let e: &MouseEvent = e.unchecked_ref();
            if e.button() == 0 { g.borrow_mut().arms.fire_held = false; }
        });
    }
    {
        let body = doc.body().ok_or("no body")?;
        let ch = doc.create_element("div")?;
        ch.set_text_content(Some("+"));
        ch.set_attribute("style", "position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);color:#fff;font:bold 22px monospace;text-shadow:0 0 3px #000;pointer-events:none;z-index:5")?;
        body.append_child(&ch)?;
        let touch = g.borrow().touch;
        if touch {
            let mk = |label: &str, css: &str| -> Result<Element, JsValue> {
                let b = doc.create_element("div")?;
                b.set_text_content(Some(label));
                b.set_attribute("style", &format!("position:fixed;{css};width:60px;height:60px;border-radius:50%;background:rgba(0,0,0,.4);color:#fff;font:24px sans-serif;display:flex;align-items:center;justify-content:center;z-index:6;user-select:none;touch-action:none"))?;
                body.append_child(&b)?;
                Ok(b)
            };
            let flyb = mk("🪽", "right:20px;top:70px")?;
            let swb = mk("🔁", "right:90px;top:70px")?;
            let rlb = mk("R", "right:160px;top:70px")?;
            let atk = mk("⚔", "right:120px;bottom:40px")?;
            { let g = g.clone(); on(&flyb, "pointerdown", move |e| { e.prevent_default(); e.stop_propagation(); g.borrow_mut().ely_toggle(); }); }
            { let g = g.clone(); on(&swb, "pointerdown", move |e| { e.prevent_default(); e.stop_propagation(); let mut gm = g.borrow_mut(); let w = (gm.arms.weapon + 1) % 3; gm.set_weapon(w); }); }
            { let g = g.clone(); on(&rlb, "pointerdown", move |e| { e.prevent_default(); e.stop_propagation(); g.borrow_mut().reload(); }); }
            { let g = g.clone(); on(&atk, "pointerdown", move |e| { e.prevent_default(); e.stop_propagation(); let mut gm = g.borrow_mut(); gm.arms.fire_held = true; gm.try_attack(); }); }
            for ev in ["pointerup", "pointercancel", "pointerleave"].iter() {
                let g = g.clone();
                on(&atk, ev, move |_| { g.borrow_mut().arms.fire_held = false; });
            }
        }
    }
      Ok(())
  }

  fn run_loop(g: Rc<RefCell<Game>>) {
      let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
      let f2 = f.clone();
      let win = g.borrow().win.clone();
      *f2.borrow_mut() = Some(Closure::new(move |t: f64| {
          let want = g.borrow_mut().frame(t);
          if want { join(&g, false); }
          if let Some(cb) = f.borrow().as_ref() { let _ = win.request_animation_frame(cb.as_ref().unchecked_ref()); }
      }));
      if let (Some(w), Some(cb)) = (web_sys::window(), f2.borrow().as_ref()) {
          let _ = w.request_animation_frame(cb.as_ref().unchecked_ref());
      };
  }

  // ---------------- Entry ----------------
  #[wasm_bindgen(start)]
  pub fn start() -> Result<(), JsValue> {
      let win = web_sys::window().ok_or("no window")?;
      let doc = win.document().ok_or("no document")?;
      let canvas: HtmlCanvasElement = el(&doc, "c")?;

      let opts = js_sys::Object::new();
      js_sys::Reflect::set(&opts, &"antialias".into(), &true.into())?;
      js_sys::Reflect::set(&opts, &"powerPreference".into(), &"high-performance".into())?;
      let gl: GL = match canvas.get_context_with_context_options("webgl2", &opts)? {
          Some(c) => c.dyn_into()?,
          None => {
              if let Some(b) = doc.body() { b.set_text_content(Some("المتصفح ما يدعم WebGL2")); }
              return Ok(());
          }
      };
      let r = Renderer::new(&gl)?;

      let name_in: HtmlInputElement = el(&doc, "name")?;
      let col_in: HtmlInputElement = el(&doc, "col")?;
      if let Ok(Some(s)) = win.local_storage() {
          if let Ok(Some(n)) = s.get_item("name") { name_in.set_value(&n); }
          match s.get_item("col") {
              Ok(Some(c)) => col_in.set_value(&c),
              _ => col_in.set_value(&format!("#{:06x}", (js_sys::Math::random() * 16_777_215.0) as u32)),
          }
      }
      let touch = win.match_media("(pointer: coarse)").ok().flatten().map(|m| m.matches()).unwrap_or(false);
      if touch { if let Some(b) = doc.body() { let _ = b.set_attribute("data-touch", "1"); } }

      let g = Rc::new(RefCell::new(Game {
          ely: Elytra { fuel: 1.0, ..Default::default() },
          arms: Arms { hp: 100, ammo: 12, weapon: 1, sent_flags: 255, ..Default::default() },
    ping: NetPing::default(),
          nav: win.navigator(),
          ui: el(&doc, "ui")?, hud: el(&doc, "hud")?, labels: el(&doc, "labels")?, chatlog: el(&doc, "chatlog")?,
          chatin: el(&doc, "chatin")?, joy: el(&doc, "joy")?, knob: el(&doc, "knob")?,
          name_in, col_in, win, doc, canvas, gl, r,
          ws: None, started: false, joined: false, my_id: 0, name: String::new(), rgb: [255; 3],
          x: 0.0, y: 0.0, z: 0.0, vy: 0.0, yaw: 0.0, face: 0.0, pitch: 0.35, dist: 7.0, ph: 0.0, amp: 0.0, grounded: true,
          keys: [false; 6], touch, chat_open: false,
          joy_id: None, joy_o: (0.0, 0.0), joy_v: (0.0, 0.0), look_id: None, look_p: (0.0, 0.0), jump_tap: false,
          infos: HashMap::new(), remotes: HashMap::new(), me_label: None,
          chat_lines: VecDeque::new(), seen_buf: Vec::with_capacity(128),
          now: 0.0, last_t: 0.0, last_send: 0.0, last_ka: 0.0, sent: [f32::NAN; 4], fps_n: 0, fps_t: 0.0, pad_on: false,
      }));
      bind(&g)?;
      run_loop(g);
      Ok(())
  }
