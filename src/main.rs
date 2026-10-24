//! سيرفر العالم: HTTP + WebSocket + Interest Management + Chat
//! بدون أي مكتبة خارجية، وملفات اللعبة مدموجة داخل الملف التنفيذي
use std::collections::HashMap;
use std::f32::consts::TAU;
use std::io::{self, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TICK: Duration = Duration::from_millis(16); // 60Hz
const MAX_CONN: usize = 1000;
const MAX_MSG: u64 = 512;
const MAX_SPEED: f32 = 12.0;
const MAX_Y: f32 = 400.0;
const VIEW: f32 = 150.0; // كل لاعب يشوف اللي ضمن هالمسافة بس
const CELL: f32 = 50.0; // حجم خلية الشبكة المكانية
const MAX_VISIBLE: usize = 128; // أقصى عدد لاعبين ينرسلون لكل لاعب
const NAME_MAX: usize = 16;
const CHAT_MAX: usize = 100;
const CHAT_GAP: Duration = Duration::from_millis(700);
const QUEUE: usize = 256;
const STACK: usize = 128 * 1024;
const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

static INDEX: &[u8] = include_bytes!("../index.html");
static JS: &[u8] = include_bytes!("../pkg/client.js");
static WASM: &[u8] = include_bytes!("../pkg/client_bg.wasm");
static CONNS: AtomicUsize = AtomicUsize::new(0);
static ETAGS: OnceLock<[String; 3]> = OnceLock::new();

type Frame = Arc<Vec<u8>>;
type Shared = Arc<Mutex<World>>;

struct Player {
    tx: SyncSender<Frame>,
    info: Option<Vec<u8>>, // [id u16, r, g, b, len, name...]
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    last: Instant,
    last_chat: Instant,
    moved: bool,
    seen: u64, // hash لآخر مجموعة لاعبين انرسلت
    c: Combat,
}
impl Player {
    fn joined(&self) -> bool { self.info.is_some() }
}

struct World {
    players: HashMap<u16, Player>,
    next_id: u16,
    dirty: bool,
}

struct ConnGuard;
impl Drop for ConnGuard {
    fn drop(&mut self) { CONNS.fetch_sub(1, Ordering::Relaxed); }
}

fn lock(w: &Shared) -> MutexGuard<'_, World> { w.lock().unwrap_or_else(|e| e.into_inner()) }

fn etags() -> &'static [String; 3] {
    ETAGS.get_or_init(|| [etag(INDEX), etag(JS), etag(WASM)])
}

fn main() {
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8080);
    let listener = TcpListener::bind(("0.0.0.0", port)).expect("bind failed");
    etags();
    println!("World on http://0.0.0.0:{port}  (wasm {} KB)", WASM.len() / 1024);

    let w: Shared = Arc::new(Mutex::new(World { players: HashMap::new(), next_id: 1, dirty: false }));
    {
        let w = w.clone();
        thread::Builder::new().name("tick".into()).spawn(move || broadcaster(w)).expect("tick");
    }
    for s in listener.incoming() {
        let Ok(s) = s else { continue };
        if CONNS.fetch_add(1, Ordering::Relaxed) >= MAX_CONN {
            CONNS.fetch_sub(1, Ordering::Relaxed);
            continue;
        }
        let w = w.clone();
        let r = thread::Builder::new().stack_size(STACK).spawn(move || {
            let _g = ConnGuard;
            let _ = handle(s, w);
        });
        if r.is_err() { CONNS.fetch_sub(1, Ordering::Relaxed); }
    }
}

// ---------------- HTTP ----------------
fn handle(mut s: TcpStream, w: Shared) -> io::Result<()> {
    s.set_nodelay(true)?;
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    s.set_write_timeout(Some(Duration::from_secs(20)))?;
    let mut buf = [0u8; 4096];
    let mut n = 0;
    loop {
        let r = s.read(&mut buf[n..])?;
        if r == 0 { return Ok(()); }
        n += r;
        if buf[..n].windows(4).any(|x| x == b"\r\n\r\n") { break; }
        if n == buf.len() { return Ok(()); }
    }
    let req = String::from_utf8_lossy(&buf[..n]).into_owned();
    let path = req.split_whitespace().nth(1).unwrap_or("/").split('?').next().unwrap_or("/").to_owned();
    let header = |name: &str| {
        req.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_owned())
        })
    };
    let inm = header("if-none-match");
    match path.as_str() {
        "/ws" => match header("sec-websocket-key") {
            Some(k) => ws_session(s, w, &k),
            None => reply(&mut s, "400 Bad Request"),
        },
        "/" | "/index.html" => serve(&mut s, 0, INDEX, "text/html; charset=utf-8", inm),
        "/pkg/client.js" => serve(&mut s, 1, JS, "text/javascript; charset=utf-8", inm),
        "/pkg/client_bg.wasm" => serve(&mut s, 2, WASM, "application/wasm", inm),
        _ => reply(&mut s, "404 Not Found"),
    }
}

fn reply(s: &mut TcpStream, status: &str) -> io::Result<()> {
    s.write_all(format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes())
}

// ETag: إذا الملف ما تغيّر يرجع 304 بدون ما يرسله مرة ثانية
fn serve(s: &mut TcpStream, i: usize, body: &[u8], ctype: &str, inm: Option<String>) -> io::Result<()> {
    let tag = &etags()[i];
    if inm.as_deref() == Some(tag.as_str()) {
        return s.write_all(format!("HTTP/1.1 304 Not Modified\r\nETag: {tag}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").as_bytes());
    }
    let h = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nETag: {tag}\r\nCache-Control: no-cache\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(h.as_bytes())?;
    s.write_all(body)
}

// ---------------- WebSocket ----------------
fn ws_session(mut s: TcpStream, w: Shared, key: &str) -> io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(90)))?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    let accept = base64(&sha1(format!("{key}{WS_GUID}").as_bytes()));
    s.write_all(format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n").as_bytes())?;

    // خيط كتابة مستقل: اللاعب البطيء ما يأخر الباقين
    let (tx, rx) = sync_channel::<Frame>(QUEUE);
    let mut out = s.try_clone()?;
    thread::Builder::new().stack_size(STACK).spawn(move || {
        for f in rx {
            if out.write_all(&f).is_err() { break; }
        }
        let _ = out.shutdown(Shutdown::Both);
    })?;

    let (id, sx, sz) = {
        let mut g = lock(&w);
        let mut id = g.next_id;
        while id == 0 || g.players.contains_key(&id) { id = id.wrapping_add(1); }
        g.next_id = id.wrapping_add(1);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7);
        let mut seed = (nanos ^ ((id as u64) << 32)) | 1;
        let a = rnd(&mut seed) * TAU;
        let r = 3.0 + rnd(&mut seed) * 10.0;
        let (sx, sz) = (a.cos() * r, a.sin() * r);
        let now = Instant::now();
        g.players.insert(id, Player {
            tx: tx.clone(), info: None, x: sx, y: 0.0, z: sz, yaw: 0.0,
            last: now, last_chat: now.checked_sub(CHAT_GAP).unwrap_or(now), moved: false, seen: 0, c: Combat::default(),
        });
        (id, sx, sz)
    };

    // [1, id, x, z]
    let mut hello = vec![1u8];
    hello.extend_from_slice(&id.to_le_bytes());
    hello.extend_from_slice(&sx.to_le_bytes());
    hello.extend_from_slice(&sz.to_le_bytes());
    let _ = tx.try_send(Arc::new(ws_frame(2, &hello)));

    let res = (|| -> io::Result<()> {
        let mut r = BufReader::with_capacity(1024, s.try_clone()?);
        let mut msg = Vec::with_capacity(128);
        loop {
            match read_frame(&mut r, &mut msg)? {
                0x8 => return Ok(()),
                0x9 => { let _ = tx.try_send(Arc::new(ws_frame(0xA, &msg))); }
                0x2 => on_msg(&w, id, &msg),
                _ => {}
            }
        }
    })();

    {
        let mut g = lock(&w);
        g.players.remove(&id);
        g.dirty = true;
        let mut b = vec![4u8];
        b.extend_from_slice(&id.to_le_bytes());
        send_all(&g, &Arc::new(ws_frame(2, &b)));
    }
    drop(tx);
    let _ = s.shutdown(Shutdown::Both);
    res
}

fn send_all(g: &World, f: &Frame) {
    for p in g.players.values() {
        if p.joined() { let _ = p.tx.try_send(f.clone()); }
    }
}

// client -> server
//   [0]                                               keepalive
//   [1, r, g, b, name...]                             دخول
//   [2, x, y, z, yaw f32, flags, pitch i8, roll i8]   حركة (آخر 3 اختيارية)
//   [3, text...]                                      شات
//   [6, weapon(|0x80 راس), target u16, dx, dy, dz]    ضربة / طلقة
//   [7]                                               ريلود
//   [12, voice...]                                    صوت
//   [13, payload...]                                  ping
//
// server -> client
//   [1, id, x, z]                                 ترحيب
//   [2, n u16, (id u16, x, y, z, yaw f32) * n]    مواقع القريبين
//   [3, (id u16, r, g, b, len, name) * ]          معلومات لاعبين
//   [4, id]                                       خروج
//   [5, id, text]                                 شات
//   [6, n u16, (id u16, flags, pitch, roll, hp)*n] حالة القريبين
//   [8, from u16, ix, iy, iz f32, hp]             انضربت (دفع + صحة)
//   [9, id u16, weapon, hit u16, ex, ey, ez f32]  أثر طلقة/ضربة لاعب ثاني
//   [10, x, z f32]                                إعادة ظهور
//   [11, killer u16, victim u16, weapon]          قتل
//   [12, id u16, voice...]                        صوت
//   [13, payload...]                              pong

const F_FLY: u8 = 1; // bit0 طيران، bit1-2 السلاح، bit3 تصويب، bit4 رفرفة
const S_RELOAD: u8 = 0x20;
const S_DEAD: u8 = 0x40;
const W_BAT: u8 = 1;
const W_GUN: u8 = 2;
const JUMP_Y: f32 = 4.0;
const FLY_SPEED: f32 = 60.0;
const KNOCK_SPEED: f32 = 40.0;
const KNOCK_TIME: Duration = Duration::from_millis(1500);
const EYE: f32 = 1.5;
const MAG: u8 = 12;
const RELOAD: Duration = Duration::from_millis(2200);
const RESPAWN: Duration = Duration::from_secs(3);
const BAT_RANGE: f32 = 3.2;
const BAT_GAP: Duration = Duration::from_millis(550);
const BAT_DMG: u8 = 15;
const BAT_PUSH: f32 = 22.0;
const BAT_UP: f32 = 9.0;
const GUN_RANGE: f32 = 150.0;
const GUN_GAP: Duration = Duration::from_millis(140);
const GUN_DMG: u8 = 20;
const GUN_DMG_HEAD: u8 = 50;
const GUN_PUSH: f32 = 2.0;
const VOICE_DIST: f32 = 40.0;
const VOICE_RATE: f32 = 20000.0; // بايت/ثانية لكل لاعب

struct Combat {
    hp: u8,
    flags: u8,
    pitch: i8,
    roll: i8,
    ammo: u8,
    reload_until: Option<Instant>,
    next_shot: Instant,
    knock_until: Instant,
    dead_until: Option<Instant>,
    frozen_until: Instant,
    voice_t: Instant,
    voice_tokens: f32,
    ts: u32,
}
impl Default for Combat {
    fn default() -> Self {
        let n = Instant::now();
        Combat {
            hp: 100, flags: 0, pitch: 0, roll: 0, ammo: MAG, reload_until: None,
            next_shot: n, knock_until: n, dead_until: None, frozen_until: n,
            voice_t: n, voice_tokens: VOICE_RATE * 0.5, ts: 0,
        }
    }
}

fn f32at(m: &[u8], i: usize) -> f32 { f32::from_le_bytes([m[i], m[i + 1], m[i + 2], m[i + 3]]) }

fn send_near(g: &World, x: f32, z: f32, r: f32, skip: u16, f: &Frame) {
    for (&oid, o) in &g.players {
        if oid == skip || !o.joined() { continue; }
        let (dx, dz) = (o.x - x, o.z - z);
        if dx * dx + dz * dz <= r * r { let _ = o.tx.try_send(f.clone()); }
    }
}

fn on_msg(w: &Shared, id: u16, m: &[u8]) {
    let mut guard = lock(w);
    let g = &mut *guard;
    let now = Instant::now();
    match m.first().copied() {
        Some(1) if m.len() >= 4 => {
            let Some(p) = g.players.get_mut(&id) else { return };
            if p.joined() { return; }
            let mut name = clean(&m[4..], NAME_MAX);
            if name.is_empty() { name = format!("Player{id}"); }
            let mut e = Vec::with_capacity(6 + name.len());
            e.extend_from_slice(&id.to_le_bytes());
            e.extend_from_slice(&m[1..4]);
            e.push(name.len() as u8);
            e.extend_from_slice(name.as_bytes());
            p.info = Some(e.clone());
            p.moved = true;
            p.last = now;
            let me_tx = p.tx.clone();
            g.dirty = true;
            let mut all = vec![3u8];
            for o in g.players.values() {
                if let Some(i) = &o.info { all.extend_from_slice(i); }
            }
            let _ = me_tx.try_send(Arc::new(ws_frame(2, &all)));
            let mut one = vec![3u8];
            one.extend_from_slice(&e);
            let f = Arc::new(ws_frame(2, &one));
            for (&oid, o) in &g.players {
                if oid != id && o.joined() { let _ = o.tx.try_send(f.clone()); }
            }
        }
        Some(2) if m.len() >= 17 => {
            let Some(p) = g.players.get_mut(&id) else { return };
            if !p.joined() || now < p.c.frozen_until || p.c.hp == 0 { return; }
            let (x, y, z, yaw) = (f32at(m, 1), f32at(m, 5), f32at(m, 9), f32at(m, 13));
            if !(x.is_finite() && y.is_finite() && z.is_finite() && yaw.is_finite()) { return; }
            let (flags, pitch, roll) = if m.len() >= 20 { (m[17] & 0x1F, m[18] as i8, m[19] as i8) } else { (0, 0, 0) };
            if m.len() >= 24 { p.c.ts = u32::from_le_bytes([m[20], m[21], m[22], m[23]]); }
            let dt = now.duration_since(p.last).as_secs_f32().min(1.0);
            p.last = now;
            let flying = flags & F_FLY != 0;
            let knocked = now < p.c.knock_until;
            let lim = if flying { FLY_SPEED } else { MAX_SPEED } + if knocked { KNOCK_SPEED } else { 0.0 };
            let (dx, dz) = (x - p.x, z - p.z);
            let d = (dx * dx + dz * dz).sqrt();
            let max = lim * dt + 0.5;
            let k = if d > max { max / d } else { 1.0 };
            let ymax = if flying || knocked { MAX_Y } else { p.y.max(JUMP_Y) };
            let n = (p.x + dx * k, y.clamp(0.0, ymax), p.z + dz * k, yaw.rem_euclid(TAU));
            let st = (flags, pitch, roll);
            if n != (p.x, p.y, p.z, p.yaw) || st != (p.c.flags, p.c.pitch, p.c.roll) {
                (p.x, p.y, p.z, p.yaw) = n;
                (p.c.flags, p.c.pitch, p.c.roll) = st;
                p.moved = true;
                g.dirty = true;
            }
        }
        Some(3) if m.len() >= 2 => {
            let Some(p) = g.players.get_mut(&id) else { return };
            if !p.joined() { return; }
            if now.duration_since(p.last_chat) < CHAT_GAP { return; }
            let text = clean(&m[1..], CHAT_MAX);
            if text.is_empty() { return; }
            p.last_chat = now;
            let mut b = vec![5u8];
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(text.as_bytes());
            send_all(g, &Arc::new(ws_frame(2, &b)));
        }
        Some(6) if m.len() >= 16 => attack(g, id, m, now),
        Some(7) => {
            let Some(p) = g.players.get_mut(&id) else { return };
            if p.joined() && p.c.hp > 0 && p.c.ammo < MAG && p.c.reload_until.is_none() {
                p.c.reload_until = Some(now + RELOAD);
                p.moved = true;
                g.dirty = true;
            }
        }
        Some(12) if m.len() >= 2 => {
            let Some(p) = g.players.get_mut(&id) else { return };
            if !p.joined() { return; }
            let el = now.duration_since(p.c.voice_t).as_secs_f32();
            p.c.voice_t = now;
            p.c.voice_tokens = (p.c.voice_tokens + el * VOICE_RATE).min(VOICE_RATE * 0.5);
            let n = m.len() as f32;
            if p.c.voice_tokens < n { return; }
            p.c.voice_tokens -= n;
            let (x, z) = (p.x, p.z);
            let mut b = Vec::with_capacity(m.len() + 2);
            b.push(12);
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&m[1..]);
            send_near(g, x, z, VOICE_DIST, id, &Arc::new(ws_frame(2, &b)));
        }
        Some(13) if m.len() <= 32 => {
            if let Some(p) = g.players.get(&id) { let _ = p.tx.try_send(Arc::new(ws_frame(2, m))); }
        }
        _ => {}
    }
}

fn attack(g: &mut World, id: u16, m: &[u8], now: Instant) {
    let wpn = m[1] & 0x7F;
    let head = m[1] & 0x80 != 0;
    let target = u16::from_le_bytes([m[2], m[3]]);
    let (mut dx, mut dy, mut dz) = (f32at(m, 4), f32at(m, 8), f32at(m, 12));
    let l = (dx * dx + dy * dy + dz * dz).sqrt();
    if !l.is_finite() || l < 0.5 { return; }
    dx /= l; dy /= l; dz /= l;
    let (range, gap, dmg, push, up) = match wpn {
        W_BAT => (BAT_RANGE, BAT_GAP, BAT_DMG, BAT_PUSH, BAT_UP),
        W_GUN => (GUN_RANGE, GUN_GAP, if head { GUN_DMG_HEAD } else { GUN_DMG }, GUN_PUSH, 0.0),
        _ => return,
    };
    let (sx, sy, sz) = {
        let Some(p) = g.players.get_mut(&id) else { return };
        if !p.joined() || p.c.hp == 0 || now < p.c.next_shot { return; }
        if wpn == W_GUN {
            if let Some(t) = p.c.reload_until {
                if now < t { return; }
                p.c.ammo = MAG;
                p.c.reload_until = None;
            }
            if p.c.ammo == 0 { return; }
            p.c.ammo -= 1;
        }
        p.c.next_shot = now + gap;
        (p.x, p.y + EYE, p.z)
    };
    let mut hit = 0u16;
    if target != 0 && target != id {
        if let Some(t) = g.players.get(&target) {
            if t.joined() && t.c.hp > 0 {
                let (vx, vy, vz) = (t.x - sx, t.y + 1.0 - sy, t.z - sz);
                let along = vx * dx + vy * dy + vz * dz;
                let (px, py, pz) = (vx - dx * along, vy - dy * along, vz - dz * along);
                let perp = (px * px + py * py + pz * pz).sqrt();
                let tol = if wpn == W_BAT { 2.5 } else { 2.0 + along * 0.02 };
                if along > -1.0 && along <= range + 2.0 && perp <= tol { hit = target; }
            }
        }
    }
    let end = match g.players.get(&hit) {
        Some(t) if hit != 0 => [t.x, t.y + if head { 1.65 } else { 1.1 }, t.z],
        _ => [sx + dx * range, sy + dy * range, sz + dz * range],
    };
    let mut fx = vec![9u8];
    fx.extend_from_slice(&id.to_le_bytes());
    fx.push(m[1]);
    fx.extend_from_slice(&hit.to_le_bytes());
    for v in end { fx.extend_from_slice(&v.to_le_bytes()); }
    send_near(g, sx, sz, VIEW, id, &Arc::new(ws_frame(2, &fx)));
    if hit == 0 { return; }

    let Some(t) = g.players.get_mut(&hit) else { return };
    let (hx, hz) = (t.x - sx, t.z - sz);
    let h = (hx * hx + hz * hz).sqrt();
    let (nx, nz) = if h > 0.05 { (hx / h, hz / h) } else { (dx, dz) };
    let imp = [nx * push, up + dy.max(0.0) * push * 0.3, nz * push];
    t.c.hp = t.c.hp.saturating_sub(dmg);
    t.c.knock_until = now + KNOCK_TIME;
    t.moved = true;
    let mut b = vec![8u8];
    b.extend_from_slice(&id.to_le_bytes());
    for v in imp { b.extend_from_slice(&v.to_le_bytes()); }
    b.push(t.c.hp);
    let _ = t.tx.try_send(Arc::new(ws_frame(2, &b)));
    let dead = t.c.hp == 0;
    if dead { t.c.dead_until = Some(now + RESPAWN); }
    g.dirty = true;
    if dead {
        let mut k = vec![11u8];
        k.extend_from_slice(&id.to_le_bytes());
        k.extend_from_slice(&hit.to_le_bytes());
        k.push(wpn);
        send_all(g, &Arc::new(ws_frame(2, &k)));
    }
}

fn respawn(p: &mut Player, now: Instant, seed: &mut u64) {
    let a = rnd(seed) * TAU;
    let r = 3.0 + rnd(seed) * 10.0;
    (p.x, p.y, p.z) = (a.cos() * r, 0.0, a.sin() * r);
    p.c.hp = 100;
    p.c.ammo = MAG;
    p.c.reload_until = None;
    p.c.dead_until = None;
    p.c.frozen_until = now + Duration::from_millis(300);
    p.moved = true;
    let mut b = vec![10u8];
    b.extend_from_slice(&p.x.to_le_bytes());
    b.extend_from_slice(&p.z.to_le_bytes());
    let _ = p.tx.try_send(Arc::new(ws_frame(2, &b)));
}

fn state_of(p: &Player) -> u8 {
    p.c.flags | if p.c.reload_until.is_some() { S_RELOAD } else { 0 } | if p.c.hp == 0 { S_DEAD } else { 0 }
}

fn clean(b: &[u8], max: usize) -> String {
    let s: String = String::from_utf8_lossy(b).chars().filter(|c| !c.is_control()).take(max).collect();
    s.trim().to_owned()
}

fn broadcaster(w: Shared) {
    let mut grid: HashMap<(i32, i32), Vec<u16>> = HashMap::new();
    let mut vis: Vec<(f32, u16)> = Vec::new();
    let mut seen: Vec<(u16, u64)> = Vec::new();
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut sbuf: Vec<u8> = Vec::with_capacity(1024);
    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(7) | 1;
    let r = (VIEW / CELL).ceil() as i32;
    loop {
        let t = Instant::now();
        {
            let mut guard = lock(&w);
            let g = &mut *guard;
            for p in g.players.values_mut() {
                if let Some(d) = p.c.dead_until {
                    if d <= t { respawn(p, t, &mut seed); g.dirty = true; }
                }
                if let Some(rl) = p.c.reload_until {
                    if rl <= t { p.c.ammo = MAG; p.c.reload_until = None; p.moved = true; g.dirty = true; }
                }
            }
            if g.dirty {
                g.dirty = false;
                grid.clear();
                for (&id, p) in &g.players {
                    if p.joined() { grid.entry(cell(p.x, p.z)).or_default().push(id); }
                }
                seen.clear();
                for (&rid, rp) in &g.players {
                    if !rp.joined() { continue; }
                    vis.clear();
                    let (cx, cz) = cell(rp.x, rp.z);
                    for gx in cx - r..=cx + r {
                        for gz in cz - r..=cz + r {
                            let Some(ids) = grid.get(&(gx, gz)) else { continue };
                            for &oid in ids {
                                if oid == rid { continue; }
                                let o = &g.players[&oid];
                                let (dx, dz) = (o.x - rp.x, o.z - rp.z);
                                let d2 = dx * dx + dz * dz;
                                if d2 <= VIEW * VIEW { vis.push((d2, oid)); }
                            }
                        }
                    }
                    if vis.len() > MAX_VISIBLE {
                        vis.select_nth_unstable_by(MAX_VISIBLE, |a, b| a.0.total_cmp(&b.0));
                        vis.truncate(MAX_VISIBLE);
                    }
                    let mut h: u64 = 0;
                    let mut any = false;
                    for &(_, oid) in &vis {
                        h = h.wrapping_add(mix(oid));
                        any |= g.players[&oid].moved;
                    }
                    if !any && h == rp.seen { continue; }
                    seen.push((rid, h));
                    buf.clear();
                    buf.push(2);
                    buf.extend_from_slice(&(vis.len() as u16).to_le_bytes());
                    sbuf.clear();
                    sbuf.push(14);
                    sbuf.extend_from_slice(&(vis.len() as u16).to_le_bytes());
                    for &(_, oid) in &vis {
                        let o = &g.players[&oid];
                        buf.extend_from_slice(&oid.to_le_bytes());
                        for v in [o.x, o.y, o.z, o.yaw] { buf.extend_from_slice(&v.to_le_bytes()); }
                        sbuf.extend_from_slice(&oid.to_le_bytes());
                        for v in [o.x, o.y, o.z, o.yaw] { sbuf.extend_from_slice(&v.to_le_bytes()); }
                        sbuf.extend_from_slice(&o.c.ts.to_le_bytes());
                        sbuf.extend_from_slice(&[state_of(o), o.c.pitch as u8, o.c.roll as u8, o.c.hp]);
                    }
                    let _ = rp.tx.try_send(Arc::new(ws_frame(2, &sbuf)));
                }
                for &(id, h) in &seen {
                    if let Some(p) = g.players.get_mut(&id) { p.seen = h; }
                }
                for p in g.players.values_mut() { p.moved = false; }
            }
        }
        thread::sleep(TICK.saturating_sub(t.elapsed()));
    }
}

fn cell(x: f32, z: f32) -> (i32, i32) { ((x / CELL).floor() as i32, (z / CELL).floor() as i32) }

fn mix(id: u16) -> u64 {
    let mut x = (id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x
}

fn read_frame<R: Read>(r: &mut R, buf: &mut Vec<u8>) -> io::Result<u8> {
    let mut h = [0u8; 2];
    r.read_exact(&mut h)?;
    let op = h[0] & 0x0F;
    let masked = h[1] & 0x80 != 0;
    let mut len = (h[1] & 0x7F) as u64;
    if len == 126 {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        len = u16::from_be_bytes(b) as u64;
    } else if len == 127 {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        len = u64::from_be_bytes(b);
    }
    if !masked || len > MAX_MSG {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad frame"));
    }
    let mut mask = [0u8; 4];
    r.read_exact(&mut mask)?;
    buf.resize(len as usize, 0);
    r.read_exact(&mut buf[..])?;
    for (i, b) in buf.iter_mut().enumerate() { *b ^= mask[i & 3]; }
    Ok(op)
}


fn ws_frame(op: u8, p: &[u8]) -> Vec<u8> {
  let mut f = Vec::with_capacity(p.len() + 10);
  f.push(0x80 | op);
  match p.len() {
      n if n < 126 => f.push(n as u8),
      n if n < 65536 => { f.push(126); f.extend_from_slice(&(n as u16).to_be_bytes()); }
      n => { f.push(127); f.extend_from_slice(&(n as u64).to_be_bytes()); }
  }
  f.extend_from_slice(p);
  f
}

fn rnd(s: &mut u64) -> f32 {
  *s ^= *s << 13;
  *s ^= *s >> 7;
  *s ^= *s << 17;
  (*s >> 40) as f32 / 16_777_216.0
}

// ETag: بصمة للملف تنحسب مرة وحدة عند التشغيل (FNV-1a)
fn etag(b: &[u8]) -> String {
  let mut h: u64 = 0xcbf2_9ce4_8422_2325;
  for &x in b { h ^= x as u64; h = h.wrapping_mul(0x0100_0000_01b3); }
  format!("\"{h:016x}-{}\"", b.len())
}

fn sha1(data: &[u8]) -> [u8; 20] {
  let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
  let mut m = data.to_vec();
  let bits = (data.len() as u64).wrapping_mul(8);
  m.push(0x80);
  while m.len() % 64 != 56 { m.push(0); }
  m.extend_from_slice(&bits.to_be_bytes());
  for c in m.chunks_exact(64) {
      let mut w = [0u32; 80];
      for i in 0..16 { w[i] = u32::from_be_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]); }
      for i in 16..80 { w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1); }
      let [mut a, mut b, mut cc, mut d, mut e] = h;
      for (i, wi) in w.iter().enumerate() {
          let (f, k) = match i {
              0..=19 => ((b & cc) | (!b & d), 0x5A827999),
              20..=39 => (b ^ cc ^ d, 0x6ED9EBA1),
              40..=59 => ((b & cc) | (b & d) | (cc & d), 0x8F1BBCDC),
              _ => (b ^ cc ^ d, 0xCA62C1D6u32),
          };
          let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
          e = d; d = cc; cc = b.rotate_left(30); b = a; a = t;
      }
      for (x, v) in h.iter_mut().zip([a, b, cc, d, e]) { *x = x.wrapping_add(v); }
  }
  let mut out = [0u8; 20];
  for (i, v) in h.iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes()); }
  out
}

fn base64(d: &[u8]) -> String {
  const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let mut o = String::with_capacity((d.len() + 2) / 3 * 4);
  for c in d.chunks(3) {
      let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
      let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
      o.push(T[(n >> 18 & 63) as usize] as char);
      o.push(T[(n >> 12 & 63) as usize] as char);
      o.push(if c.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
      o.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
  }
  o
}

