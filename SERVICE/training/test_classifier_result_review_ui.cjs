// Mock DOM/API fixtures only; never relabel the real dataset during UI tests.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Element{
 constructor(tag='DIV'){this.tagName=tag;this.children=[];this.value='';this.textContent='';this.className='';this.dataset={};this.handlers={};this.disabled=false;this.classList={toggle(){}};}
 append(...nodes){for(const node of nodes){node.parentNode=this;this.children.push(node);}}
 replaceChildren(...nodes){this.children=[];this.append(...nodes);}
 setAttribute(){}
 addEventListener(name,fn){this.handlers[name]=fn;}
 querySelectorAll(selector){const all=[];function visit(node){for(const child of node.children){all.push(child);visit(child);}}visit(this);return all.filter(node=>selector==='.labels button'?node.tagName==='BUTTON'&&node.parentNode.className==='labels':false);}
 querySelector(selector){return this.children.find(node=>node.className.split(' ').includes(selector.slice(1)));}
}
const elements={},document={body:new Element(),getElementById(id){return elements[id]??=new Element();},createElement(tag){return new Element(tag.toUpperCase());},querySelectorAll(){return elements.grid?.querySelectorAll('.labels button')||[];}};
for(const [id,value] of [['mode','new'],['domain',''],['review',''],['label',''],['sort','asc']])document.getElementById(id).value=value;
const items=[['a','ng','ng','ok',.95],['b','ok','ok','ng',.4],['c','no_disk','ok','ok',.7],['d','ok','ng','ok',.6]].map(([id,gt,old,now,score])=>({id,original_label:gt,domain:'ha_test',image:id+'.png',path:'fixture',predictions:{baseline:{label:old,confidence:.8},candidate:{label:now,confidence:score,probabilities:{ng:.2,ok:.4,no_disk:.4}}}}));
let stored={revision:0,decisions:{}},saved=[],fail=false;
async function fetchMock(url,args){if(url==='/api/review')return{ok:true,json:async()=>({items,state:structuredClone(stored),evaluation_sha256:'fixture',report:{domains:{}}})};
 const body=JSON.parse(args.body);saved.push(body);if(fail)return{ok:false,json:async()=>({error:'fixture save rejected'})};
 assert.equal(body.revision,stored.revision);assert.equal(body.evaluation_sha256,'fixture');stored.revision++;
 for(const change of body.decisions){if(change.action==='reset')delete stored.decisions[change.id];else stored.decisions[change.id]={label:change.label,notes:change.notes,original_label:items.find(i=>i.id===change.id).original_label};}
 return{ok:true,json:async()=>({state:structuredClone(stored)})};}
const context=vm.createContext({document,fetch:fetchMock,structuredClone,console});
const script=fs.readFileSync(__dirname+'/classifier_result_review.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];vm.runInContext(script,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{await tick();
 assert.equal(elements.grid.children.length,3);assert.equal(elements.grid.children[0].children[0].children[0].alt,'b');
 const first=elements.grid.children[0],labels=first.children.find(c=>c.className==='labels');
 assert.match(first.children.find(c=>c.className==='prediction').textContent,/0.400000/);
 await labels.children[1].onclick();assert.equal(saved[0].decisions[0].id,'b');assert.equal(saved[0].decisions[0].label,'ok');
 assert.equal(stored.decisions.b.label,'ok');assert.equal(items[1].original_label,'ok');assert.equal(elements.grid.children.length,3); // no vanishing after correction
 assert.match(first.querySelector('.current').textContent,/ok/);assert.match(first.querySelector('.feedback').textContent,/Đã lưu/);
 elements.review.value='changed';elements.review.handlers.change();assert.equal(elements.grid.children.length,0);
 elements.review.value='';elements.mode.value='all';elements.mode.handlers.change();assert.equal(elements.grid.children.length,4);
 elements.mode.value='new';elements.mode.handlers.change();const card=elements.grid.children[0],buttons=card.children.find(c=>c.className==='labels');
 await buttons.children[0].onclick();assert.equal(stored.decisions.b.label,'ng');assert.equal(card.className,'changed');
 await buttons.children[3].onclick();assert.equal(stored.decisions.b,undefined);assert.equal(card.className,'');
 fail=true;await buttons.children[2].onclick();assert.equal(stored.decisions.b,undefined);assert.match(card.querySelector('.feedback').textContent,/CHƯA LƯU/);assert.equal(buttons.children[0].disabled,false);
 assert.match(elements.status.textContent,/CHƯA LƯU/);
 console.log('Result review UI fixtures PASS: confidence order, stable cards, correct IDs, save/reset, failure feedback, immutable GT.');
})().catch(error=>{console.error(error);process.exitCode=1;});
