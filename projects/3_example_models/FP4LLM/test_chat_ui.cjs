const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('web/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
function element(){return {value:'',style:{},children:[],append(...nodes){for(const n of nodes)n.parentElement=this;this.children.push(...nodes)},querySelector(){return this.children[0]},focus(){}}}
async function test(initialActive){
  const ids = Object.fromEntries(['welcome','conversation','scroll','send','mode','new','prompt','status','warning'].map(id=>[id,element()]));
  ids.mode.value='fpga';ids.prompt.value='Keep my draft';
  let reads=0,posts=0;
  const active={status:'loading',mode:'fpga',text:'',detail:'FPGA working'};
  const done={status:'done',mode:'fpga',text:'Hello!',detail:'Complete'};
  const sandbox={document:{getElementById:id=>ids[id],createElement:element,querySelectorAll:()=>[]},setTimeout:fn=>setImmediate(fn),window:{}};
  sandbox.window.fetch=async url=>{
    if(url==='/api/chat'){posts++;return {ok:false,status:409,json:async()=>({error:'A response is already running'})}}
    reads++;
    const s=initialActive?(reads===1?active:done):(reads<=2?{status:'idle'}:reads===3?active:done);
    return {ok:true,json:async()=>s};
  };
  sandbox.fetch=(...args)=>sandbox.window.fetch(...args);
  vm.createContext(sandbox);vm.runInContext(script,sandbox);
  for(let n=0;n<10;n++)await new Promise(setImmediate);
  if(!initialActive)await vm.runInContext('send()',sandbox);
  assert.equal(ids.prompt.value,'Keep my draft');
  assert.equal(vm.runInContext('busy',sandbox),false);
  assert.equal(vm.runInContext('history.length',sandbox),0);
  assert.equal(posts,initialActive?0:1);
  assert.equal(ids.conversation.children.at(-1).children[1].textContent,'Hello!');
}
(async()=>{await test(true);await test(false);console.log('PASS: reconnect and 409 race show active response, preserve draft, and avoid mismatched history.')})().catch(e=>{console.error(e);process.exitCode=1});
