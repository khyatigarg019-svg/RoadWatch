
const KEY='roadwatch_reports';
function reports(){return JSON.parse(localStorage.getItem(KEY)||'[]')}
function save(rows){localStorage.setItem(KEY,JSON.stringify(rows))}
function user(){return localStorage.getItem('roadwatch_user')||'Guest'}
function esc(s=''){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
