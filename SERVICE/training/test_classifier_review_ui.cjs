// No browser required: fixture-only tests of UI sorting, score display and save IDs.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { webcrypto } = require('node:crypto');
class Element {
  constructor(tag='DIV') {this.tagName=tag;this.value='';this.children=[];this.textContent='';this.style={};this.handlers={};}
  append(...values) {this.children.push(...values);}
  add(value) {this.append(value);}
  replaceChildren(...values) {this.children=values;}
  setAttribute() {}
  addEventListener(type, action) {this.handlers[type]=action;}
  querySelectorAll() {return this.children;}
  showModal() {this.open=true;}
  close() {this.open=false;}
}
const elements={};
const doc={getElementById(id){return elements[id]??=new Element();},createElement(tag){return new Element(tag.toUpperCase());},addEventListener(){},activeElement:new Element()};
doc.getElementById('status').value='unreviewed';
doc.getElementById('limit').value='1';
doc.getElementById('sort').value='asc';
const items=[['a','ng',.95],['b','ok',.4],['c','no_disk',.7]].map(([id,label,confidence])=>({id,frame:'frame',camera:1,region:'row1_bottom',slot:1,shape:[165,114,3],image:id+'.png',context:id+'.png',proposed_label:label,confidence,probabilities:{ng:.3,no_disk:.3,ok:.4}}));
const bytes=Buffer.from(JSON.stringify({items,class_counts:{ng:1,ok:1,no_disk:1}}));
let state={decisions:{},training_authorized:false},saved=[];
async function fetchMock(url, args) {
 if(url==='items.json')return {ok:true,arrayBuffer:async()=>bytes};
 if(url==='/api/state')return {ok:true,json:async()=>({ ...state, items_sha256:Buffer.from(await webcrypto.subtle.digest('SHA-256',bytes)).toString('hex') })};
 if(url==='/api/save') {const body=JSON.parse(args.body);saved=body.decisions;for(const d of saved)state.decisions[d.id]={label:d.label,notes:d.notes};return {ok:true,json:async()=>({saved:saved.length,state})};}
 throw Error(url);
}
const context=vm.createContext({document:doc,fetch:fetchMock,crypto:webcrypto,TextDecoder,Uint8Array,console,Option:function(text,value){this.textContent=text;this.value=value;},confirm:()=>true});
const script=fs.readFileSync(__dirname+'/classifier_label_review.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInContext(script,context);
async function settle(){await new Promise(r=>setTimeout(r,40));}
(async()=>{
 await settle();
 assert.equal(elements.grid.children.length,3);
 assert.equal(elements.grid.children[0].children[0].alt,'b');
 assert.equal(elements.grid.children[0].children[1].textContent,'ok · 0.400000');
 elements.label.value='no_disk';elements.label.handlers.change();
 assert.equal(elements.grid.children.length,1);assert.equal(elements.grid.children[0].children[0].alt,'c');
 elements.label.value='';elements.label.handlers.change();
 const selector=elements.grid.children[0].children[3];selector.value='ng';await selector.onchange();
 assert.equal(saved[0].id,'b');assert.equal(saved[0].label,'ng');
 assert.equal(items[1].proposed_label,'ok');assert.equal(items[1].confidence,.4);
 assert.equal(elements.grid.children.length,2);
 elements.status.value='confirmed';elements.status.handlers.change();
 assert.equal(elements.grid.children.length,1);
 assert.equal(elements.grid.children[0].children[1].textContent,'ok · 0.400000');
 assert.equal(elements.grid.children[0].children[3].value,'ng');
 assert.equal(state.training_authorized,false);
 console.log('UI fixture tests PASS: low-to-high, class filter, human label persistence, immutable model score.');
})().catch(error=>{console.error(error);process.exitCode=1;});
