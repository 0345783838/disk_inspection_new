// Isolated UI fixture: no writes to the user's canonical dataset.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
class Element {
  constructor(tag='DIV') { this.tagName=tag; this.value=''; this.children=[]; this.handlers={}; this.style={}; }
  append(...values) { this.children.push(...values); }
  add(value) { this.append(value); }
  get options() { return this.children; }
  replaceChildren(...values) { this.children=values; }
  setAttribute() {}
  addEventListener(type, handler) { this.handlers[type]=handler; }
}
const elements={}, created=[];
const doc={getElementById(id){return elements[id]??=new Element();},
  createElement(tag){const e=new Element(tag.toUpperCase());created.push(e);return e;},
  querySelectorAll(){return [...Object.values(elements),...created];}};
doc.getElementById('view').value='all';
const rows=[{id:'ha',origin:'old_ha',label:'ok'}, {id:'new',origin:'new_factory',label:'no_disk'},
  {id:'legacy',origin:'legacy_val',label:'ng'}].map(r=>({...r,image:r.id+'.png',camera:r.id==='ha'?1:r.id==='new'?2:null,role:r.origin==='legacy_val'?'evaluation_only':'training_pool'}));
const predictions={ha:{label:'ng',confidence:.8,probabilities:{ng:.8,ok:.1,no_disk:.1}},
  new:{label:'no_disk',confidence:.9,probabilities:{ng:.05,ok:.05,no_disk:.9}}};
let revision=8, requests=[];
async function fetchMock(url,args) {
  if(url==='/api/state')return {ok:true,json:async()=>({revision,records:rows,pairs:[],dataset_path:'fixture',train_count:2,validation_count:1,excluded_count:0})};
  if(url==='/api/evaluation')return {ok:true,json:async()=>({available:true,canonical_revision:revision,by_id:predictions,counts_by_origin:{},missing_prediction_ids:[]})};
  if(url==='/api/labels') {
    const body=JSON.parse(args.body);requests.push(body);assert.equal(body.revision,revision);
    const {id,label}=body.changes[0], r=rows.find(r=>r.id===id);r.label=label;if(label==='uncertain')r.role='excluded';revision++;
    return {ok:true,json:async()=>({revision,updated:[r],train_count:2,validation_count:1,excluded_count:label==='uncertain'?1:0,counts:{},saved:1})};
  }
  throw Error(url);
}
const context=vm.createContext({document:doc,fetch:fetchMock,URLSearchParams,location:{search:''},
  history:{replaceState(){}},Option:function(text,value){this.textContent=text;this.value=value;},console});
const html=fs.readFileSync(__dirname+'/classifier_canonical_review.html','utf8');
assert.ok(html.indexOf('option value="all"')<html.indexOf('option value="pairs"'));
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const settle=()=>new Promise(resolve=>setTimeout(resolve,20));
const cards=()=>elements.content.children[0].children;
function set(id,value){elements[id].value=value;elements[id].handlers.change();}
(async()=>{
  await settle();
  assert.equal(cards().length,3);
  assert.equal(cards()[0].children[2].value,'ok'); // Current label, despite predicted NG.
  assert.equal(cards()[0].children.length,5); // Model predictions hidden in full review.
  set('camera','1');assert.equal(cards().length,1);assert.equal(cards()[0].children[2].value,'ok');
  set('camera','2');assert.equal(cards().length,1);assert.equal(cards()[0].children[2].value,'no_disk');
  set('camera','unknown');assert.equal(cards().length,1);assert.equal(cards()[0].children[2].value,'ng');
  set('camera','');
  set('split','val');assert.equal(cards().length,1);assert.equal(cards()[0].children[2].value,'ng');
  set('split','train');assert.equal(cards().length,2);
  set('split','');
  set('view','evaluation');assert.equal(cards().length,2);
  set('errorOnly','different');assert.equal(cards().length,1);
  const selector=cards()[0].children[2];selector.value='ng';await selector.onchange();
  assert.equal(requests[0].changes[0].id,'ha');assert.equal(cards().length,0);
  set('errorOnly','');
  await cards()[1].children[4].onclick(); // Explicit confirmation saves current no_disk.
  assert.equal(requests[1].changes[0].label,'no_disk');
  const exclude=cards()[1].children[2];exclude.value='uncertain';await exclude.onchange();
  assert.equal(cards().length,1); // Exclusions leave diagnostic cohort.
  set('view','all');assert.equal(cards().length,3); // Still available in full review.
  console.log('Canonical UI PASS: full review, current labels, HA/new scope, disagreement filters, confirm/save, exclusions.');
})().catch(error=>{console.error(error);process.exitCode=1;});
