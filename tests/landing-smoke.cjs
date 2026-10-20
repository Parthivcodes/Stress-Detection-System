// Behavioral smoke checks. Browser QA covers the real WebGL renderer and layout.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, '../frontend/landing.js'), 'utf8');
function boot({reduce=false,mobile=false,webgl=true}={}) {
  const elements = new Map(), frames = new Map(); let nextFrame=0, geometry;
  class Element {
    constructor(){this.handlers={};this.attributes={};this.style={};this.dataset={};this.textContent='';this.clientWidth=mobile?390:1194;this.clientHeight=mobile?838:789;this.classList={add(){},remove(){}};this.firstElementChild={textContent:''};}
    addEventListener(name,fn){this.handlers[name]=fn;}
    setAttribute(name,value){this.attributes[name]=value;}
    append(){} focus(){} showModal(){this.open=true;} close(){this.open=false;}
    getBoundingClientRect(){return {left:0,top:0,width:this.clientWidth,height:this.clientHeight};}
    getContext(){return {scale(){},clearRect(){},fillRect(){}};}
    click(){this.handlers.click?.({});}
  }
  const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const buttons=Array.from({length:5},(_,i)=>{const e=new Element();e.dataset.form=String(i);return e;});
  const doc={getElementById:get,querySelectorAll:()=>buttons,querySelector:()=>get('selectors'),addEventListener(){},hidden:false};
  const THREE={
    WebGLRenderer:class{constructor(){this.domElement=new Element();}setClearColor(){}setSize(){}setPixelRatio(){}render(){}dispose(){}},
    Scene:class{add(){}},PerspectiveCamera:class{constructor(){this.position={};}updateProjectionMatrix(){}},
    BufferGeometry:class{constructor(){this.attributes={};geometry=this;}setAttribute(k,v){this.attributes[k]=v;}},
    BufferAttribute:class{constructor(array,size){this.array=array;this.itemSize=size;}},
    ShaderMaterial:class{constructor(options){Object.assign(this,options);}},Vector2:class{set(){}},
    Points:class{constructor(){this.position={};this.rotation={};this.scale={setScalar(){}};}}
  };
  const media={matches:reduce,addEventListener(){}};
  vm.runInNewContext(code,{document:doc,window:{THREE:webgl?THREE:undefined},matchMedia:q=>q.includes('reduced')?media:{matches:mobile},devicePixelRatio:1,Float32Array,Math,ResizeObserver:class{observe(){}},setTimeout:fn=>{fn();return 1;},clearTimeout(){},requestAnimationFrame:fn=>{const id=++nextFrame;frames.set(id,fn);return id;},cancelAnimationFrame:id=>frames.delete(id)});
  const tick=now=>{const callbacks=[...frames.values()];frames.clear();callbacks.forEach(fn=>fn(now));};
  return {get,buttons,frames,tick,geometry};
}
const reduced=boot({reduce:true});
assert.equal(reduced.get('particle-motion').attributes['aria-label'],'Play animation');
assert.equal(reduced.frames.size,0,'Reduced motion must not start an animation loop');
reduced.buttons[0].click();
assert.equal(reduced.get('form-name').textContent,'Sphere');
assert.equal(reduced.geometry.attributes.aTarget.array.length,22000*3);
assert.ok([...reduced.geometry.attributes.aTarget.array].every(Number.isFinite));
const live=boot();live.tick(100);live.tick(150);live.buttons[3].click();live.tick(200);
live.buttons[4].click();live.tick(250);
assert.ok([...live.geometry.attributes.position.array].every(Number.isFinite),'Interrupted morph remains finite');
live.get('particle-motion').click();live.tick(300);
assert.equal(live.frames.size,0,'Pause stops the render loop');
live.get('particle-motion').click();assert.equal(live.frames.size,1);
assert.equal(boot({mobile:true}).geometry.attributes.position.array.length,13000*3);
const fallback=boot({webgl:false});
assert.match(fallback.get('render-status').textContent,/Static preview/);
assert.equal(fallback.get('particle-motion').disabled,true);
fallback.buttons[1].click();assert.equal(fallback.get('form-name').textContent,'Diamond');
assert.equal(fallback.frames.size,0);
console.log('PASS: reduced motion, manual forms, interrupted morphs, pause/resume, mobile budget, and no-WebGL fallback');
