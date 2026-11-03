import * as THREE from './vendor/three.module.min.js';

const $ = s => document.querySelector(s);
const clamp = THREE.MathUtils.clamp;
const debug = new URLSearchParams(location.search).get('debug') === '1';
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
async function loadData(id, url) {
  const embedded = JSON.parse($(id).textContent);
  if (embedded) return embedded;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}
const [archive, configurations] = await Promise.all([
  loadData('#archive-data', 'archive.json'), loadData('#projection-data', 'projections.json')
]);
const configMap = new Map(configurations.map(c => [c.id, c]));
const spatial = archive.photos.filter(p => p.spatial);
const buttons = new Map();
let current, currentConfig, activeButton, generation = 0, currentTexture, loadingPhoto=false;
let frameProgress = 0, dirty = true, disposed = false;
let travel=0, targetTravel=0, travelVelocity=0, settleTimer;
let entrance=0, pointerX=0, pointerY=0, lookX=0, lookY=0, lastFrame=0;
const photoIndex=new Map(archive.photos.map((p,i)=>[p.id,i]));
const frame = $('#picture-frame');
const canvas = $('#scene');
const fallback = $('#fallback');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, .01, 1000);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setClearColor(0x101010, 1);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
} catch (error) {
  console.error('WebGL unavailable; original photographs remain available.', error);
  canvas.hidden = true;
}
const room = new THREE.Group();
scene.add(room);
const loader = new THREE.TextureLoader();

// Eight unknowns, h33 = 1. Solve the projectivity, not a bilinear UV warp.
export function homography(from, to) {
  const a = [];
  for (let i = 0; i < 4; i++) {
    const [x,y] = from[i], [u,v] = to[i];
    a.push([x,y,1,0,0,0,-u*x,-u*y,u]);
    a.push([0,0,0,x,y,1,-v*x,-v*y,v]);
  }
  for (let c = 0; c < 8; c++) {
    let pivot = c;
    for (let r=c+1;r<8;r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot=r;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    if (Math.abs(a[c][c]) < 1e-11) throw new Error('Degenerate projection region');
    const divisor=a[c][c];
    for (let k=c;k<9;k++) a[c][k]/=divisor;
    for (let r=0;r<8;r++) if (r!==c) {
      const factor=a[r][c];
      for (let k=c;k<9;k++) a[r][k]-=factor*a[c][k];
    }
  }
  return [...a.map(row=>row[8]),1];
}
const square = [[0,0],[1,0],[1,1],[0,1]];

export function deriveRoom(config, aspect) {
  const v=config.vanishingPoint,b=config.backWall;
  if (!(0<b.left && b.left<v.x && v.x<b.right && b.right<1 &&
        0<b.top && b.top<v.y && v.y<b.bottom && b.bottom<1))
    throw new Error('VP must lie inside the back wall, which must lie inside the photograph.');
  const f=1/(2*Math.tan(THREE.MathUtils.degToRad(25)));
  // The first front rectangle covering the image is the ray-expanded back wall.
  // Overshoot is intentionally untextured: missing pixels are never clamped.
  const ratio=Math.max(v.x/(v.x-b.left),(1-v.x)/(b.right-v.x),
    v.y/(v.y-b.top),(1-v.y)/(b.bottom-v.y));
  let depth=f*(ratio-1)*clamp(config.depth ?? 1,.05,4);
  const backDistance=f+depth, scale=backDistance/f;
  const back=[[b.left,b.top],[b.right,b.top],[b.right,b.bottom],[b.left,b.bottom]];
  const outer=back.map(([x,y])=>[v.x+(x-v.x)*scale,v.y+(y-v.y)*scale]);
  const point=(p,d)=>new THREE.Vector3((p[0]-v.x)*aspect*d/f,(v.y-p[1])*d/f,-d);
  const projectedBack=back.map(p=>point(p,backDistance));
  // The photo's VP/back-wall scale establishes the size; all three room axes
  // use that size, forming an open-front cube rather than a shallow image box.
  const side=Math.max(projectedBack[0].distanceTo(projectedBack[1]),projectedBack[0].distanceTo(projectedBack[3]));
  const cx=(projectedBack[0].x+projectedBack[2].x)/2;
  const cy=(projectedBack[0].y+projectedBack[2].y)/2;
  depth=side;
  const B=[[-1,1],[1,1],[1,-1],[-1,-1]].map(([x,y])=>new THREE.Vector3(cx+x*side/2,cy+y*side/2,-f-side));
  const O=B.map(p=>new THREE.Vector3(p.x,p.y,-f));
  return {f,depth,side,cx,cy,backDistance:f+side,point,regions:{
    back:{source:back,vertices:B},
    sky:{source:[outer[0],outer[1],back[1],back[0]],vertices:[O[0],O[1],B[1],B[0]]},
    ground:{source:[back[3],back[2],outer[2],outer[3]],vertices:[B[3],B[2],O[2],O[3]]},
    leftWall:{source:[outer[0],back[0],back[3],outer[3]],vertices:[O[0],B[0],B[3],O[3]]},
    rightWall:{source:[back[1],outer[1],outer[2],back[2]],vertices:[B[1],O[1],O[2],B[2]]}
  }};
}

const vertexShader = `
  varying vec2 planeUV;
  void main(){planeUV=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}
`;
const fragmentShader = `
  uniform sampler2D photo;
  uniform mat3 projectivity;
  uniform vec2 photoSize;
  uniform vec2 feather;
  uniform float enabled;
  uniform float gradientMode;
  uniform vec3 skyTop;
  uniform vec3 skyBottom;
  uniform float darkening;
  uniform float maskRegion;
  uniform vec2 regionQuad[4];
  varying vec2 planeUV;
  void main(){
    vec2 q=vec2(planeUV.x,1.0-planeUV.y);
    vec3 projected=projectivity*vec3(q,1.0);
    if(abs(projected.z)<0.00001) discard;
    vec2 source=projected.xy/projected.z;
    float regionAlpha=1.0;
    if(maskRegion>.5){
      float distanceToEdge=100000.0;
      for(int i=0;i<4;i++){
        vec2 a=regionQuad[i]*photoSize;
        vec2 b=regionQuad[i==3?0:i+1]*photoSize;
        vec2 p=source*photoSize;
        float crossProduct=(b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x);
        distanceToEdge=min(distanceToEdge,crossProduct/max(length(b-a),.001));
      }
      if(distanceToEdge < -3.0)discard;
      regionAlpha=smoothstep(-3.0,0.0,distanceToEdge);
    }
    // Discard outside the photograph BEFORE sampling. Clamp-to-edge cannot smear.
    if(any(lessThan(source,vec2(0.0)))||any(greaterThan(source,vec2(1.0)))) discard;
    vec2 edge=min(q,1.0-q);
    float seam=smoothstep(0.0,feather.x,edge.x)*smoothstep(0.0,feather.y,edge.y);
    vec2 px=min(source,1.0-source)*photoSize;
    float border=smoothstep(0.0,4.0,min(px.x,px.y));
    vec3 ink=vec3(16.0/255.0);
    ink=mix(ink/12.92,pow((ink+0.055)/1.055,vec3(2.4)),step(vec3(.04045),ink));
    vec3 color=texture2D(photo,vec2(source.x,1.0-source.y)).rgb;
    if(gradientMode>.5) color=mix(skyTop,skyBottom,smoothstep(0.0,1.0,q.y));
    color=mix(ink,color,enabled*(1.0-darkening));
    // Original-image perimeter and newly exposed areas dissolve into #101010.
    gl_FragColor=vec4(mix(ink,color,border),seam*regionAlpha);
    #include <colorspace_fragment>
  }
`;

function sampleSky(texture) {
  const c=document.createElement('canvas');c.width=16;c.height=16;
  const ctx=c.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(texture.image,0,0,16,16);
  function row(y) {
    const data=ctx.getImageData(4,y,8,1).data;
    const sum=[0,0,0];for(let i=0;i<data.length;i+=4) for(let j=0;j<3;j++) sum[j]+=data[i+j]/(8*255);
    return new THREE.Color(...sum).convertSRGBToLinear();
  }
  return [row(0),row(3)];
}
let derived;
function rebuild() {
  if(!renderer || !currentTexture) return;
  while(room.children.length) {
    const m=room.children[0];room.remove(m);m.geometry.dispose();m.material.dispose();
  }
  if(!current.spatial){
    derived=null;
    const geometry=new THREE.PlaneGeometry(current.width/current.height,1);
    const material=new THREE.MeshBasicMaterial({map:currentTexture,side:THREE.DoubleSide});
    const mesh=new THREE.Mesh(geometry,material);mesh.name='sign';room.add(mesh);
    resize();drawDebug();dirty=true;return;
  }
  derived=deriveRoom(currentConfig,current.width/current.height);
  const skyColors=sampleSky(currentTexture);
  const {width:pw,height:ph}=current;
  for(const [name,region] of Object.entries(derived.regions)) {
    const vertices=region.vertices.map(v=>v.clone());
    const farSky=false; // The immersive room keeps the sky on its actual ceiling.
    let sourceRegion=region.source;
    if(farSky) {
      const far=derived.backDistance+derived.depth*4;
      // Use a rectangular distant plane with a source-space polygon mask.
      // A trapezoidal mesh would introduce a second projectivity in its UVs.
      square.forEach((p,i)=>vertices[i].copy(derived.point(p,far)));
      sourceRegion=square;
    }
    const width=vertices[0].distanceTo(vertices[1]),height=vertices[0].distanceTo(vertices[3]);
    const eX=new THREE.Vector3().subVectors(vertices[1],vertices[0]).multiplyScalar(3/pw);
    const eY=new THREE.Vector3().subVectors(vertices[3],vertices[0]).multiplyScalar(3/ph);
    // Expand each plane by three source pixels and extrapolate its projectivity.
    const expanded=[vertices[0].clone().sub(eX).sub(eY),vertices[1].clone().add(eX).sub(eY),
      vertices[2].clone().add(eX).add(eY),vertices[3].clone().sub(eX).add(eY)];
    const baseH=homography(square,sourceRegion);
    const at=(x,y)=>{const d=baseH[6]*x+baseH[7]*y+1;return [(baseH[0]*x+baseH[1]*y+baseH[2])/d,(baseH[3]*x+baseH[4]*y+baseH[5])/d];};
    const source=[at(-3/pw,-3/ph),at(1+3/pw,-3/ph),at(1+3/pw,1+3/ph),at(-3/pw,1+3/ph)];
    const h=homography(square,source);
    const geometry=new THREE.PlaneGeometry(width,height);
    const positions=geometry.attributes.position;
    // PlaneGeometry stores TL, TR, BL, BR.
    [0,1,3,2].forEach((corner,i)=>positions.setXYZ(i,...expanded[corner].toArray()));
    positions.needsUpdate=true;geometry.computeVertexNormals();geometry.computeBoundingSphere();
    const enabled=!(currentConfig[name]?.enabled===false);
    const material=new THREE.ShaderMaterial({vertexShader,fragmentShader,side:THREE.DoubleSide,
      transparent:true,depthWrite:true,polygonOffset:true,polygonOffsetFactor:name==='back'?-1:0,
      uniforms:{photo:{value:currentTexture},projectivity:{value:new THREE.Matrix3().set(...h)},
        photoSize:{value:new THREE.Vector2(pw,ph)},feather:{value:new THREE.Vector2(3/pw,3/ph)},
        enabled:{value:enabled?1:0},gradientMode:{value:name==='sky'&&currentConfig.sky.mode==='gradient'?1:0},
        skyTop:{value:skyColors[0]},skyBottom:{value:skyColors[1]},darkening:{value:0},
        maskRegion:{value:farSky?1:0},regionQuad:{value:region.source.map(p=>new THREE.Vector2(...p))}}});
    const mesh=new THREE.Mesh(geometry,material);mesh.name=name;mesh.renderOrder=name==='back'?2:1;
    room.add(mesh);
  }
  camera.near=Math.max(.001,derived.f*.01);camera.far=derived.backDistance+derived.depth*12+10;
  resize();drawDebug();dirty=true;
}

function resize() {
  if(!current)return;
  const stage=$('#stage'), aspect=current.width/current.height;
  const availableWidth=stage.clientWidth*(innerWidth<=700?.94:.72);
  const width=Math.min(availableWidth,innerHeight*.66*aspect);
  frame.style.width=`${width}px`;frame.style.height=`${width/aspect}px`;
  if(renderer) {renderer.setSize(width,width/aspect,false);camera.aspect=aspect;}
  dirty=true;
}
addEventListener('resize',resize);

async function selectPhoto(photo) {
  if(current?.id===photo.id)return;
  const request=++generation;loadingPhoto=true;
  current=photo;currentConfig=photo.spatial?structuredClone(configMap.get(photo.id)):null;entrance=0;
  $('#photo-year').textContent=`${photo.year}s`;
  $('#photo-description').textContent=photo.description;
  if(activeButton)activeButton.removeAttribute('aria-current');
  activeButton=buttons.get(photo.id);activeButton?.setAttribute('aria-current','true');
  updateTimeline(photo.era);resize();
  try {
    const texture=await loader.loadAsync(photo.src);
    if(request!==generation){texture.dispose();return;}
    texture.colorSpace=THREE.SRGBColorSpace;
    texture.anisotropy=renderer?Math.min(8,renderer.capabilities.getMaxAnisotropy()):1;
    currentTexture?.dispose();currentTexture=texture;
    if(renderer){fallback.hidden=!debug;canvas.hidden=debug;if(debug){fallback.src=photo.src;fallback.alt=photo.description;}rebuild();}
    else{fallback.src=photo.src;fallback.alt=photo.description;fallback.hidden=false;}
    loadingPhoto=false;document.documentElement.dataset.ready='true';
    refreshDebugPanel();
  } catch(error) {
    if(request!==generation)return;
    console.error('Photograph load failed',photo.src,error);
    fallback.src=photo.src;fallback.alt=photo.description;fallback.hidden=false;canvas.hidden=true;
  }
}
function updateTimeline(year) {
  let active;
  for(const el of $('#eras').children) {
    const yes=Number(el.dataset.year)===year;el.classList.toggle('active',yes);if(yes)active=el;
  }
  // The timeline stays physically stationary; only its active item changes.
}

for(const era of archive.eras) {
  const section=document.createElement('section');section.className='era-group';section.dataset.year=era.year;
  const label=document.createElement('h2');label.className='era-label';label.textContent=`${era.year}s`;section.append(label);
  for(const photo of archive.photos.filter(p=>p.era===era.year)) {
    const button=document.createElement('button');button.className='photo-button';button.type='button';button.dataset.id=photo.id;
    button.setAttribute('aria-label',`${photo.year}s · ${photo.description} · 선택`);
    const img=document.createElement('img');img.src=photo.src;img.alt=photo.description;
    img.width=photo.width;img.height=photo.height;img.loading='lazy';img.decoding='async';
    // A single uniform scale per thumbnail; portrait and landscape ratios stay original.
    const scale=Math.min(1,180/Math.max(photo.width,photo.height));
    img.style.width=`${photo.width*scale}px`;
    button.append(img);button.addEventListener('click',()=>navigateTo(photoIndex.get(photo.id)));
    buttons.set(photo.id,button);section.append(button);
  }
  $('#archive').append(section);
  const item=document.createElement('section');item.className='era-event';item.dataset.year=era.year;
  const heading=document.createElement('h2');heading.textContent=`${era.year}s`;
  const policy=document.createElement('p');policy.textContent=era.policy;
  item.append(heading,policy);$('#eras').append(item);
}

// Fixed, independently scrollable archive. Stage gestures never scroll it.
function navigateTo(index, immediate=false) {
  targetTravel=clamp(index,0,archive.photos.length-1);
  if(immediate||reduced){travel=targetTravel;travelVelocity=0;}
  selectPhoto(archive.photos[Math.round(targetTravel)]);
  dirty=true;
}
function onScroll(){dirty=true;}
function pushTravel(delta) {
  targetTravel=clamp(targetTravel+delta,0,archive.photos.length-1);
  clearTimeout(settleTimer);
  settleTimer=setTimeout(()=>{targetTravel=Math.round(targetTravel);dirty=true;},180);
  dirty=true;
}
$('#stage').addEventListener('wheel',e=>{
  if(debug||dialog.open)return;
  e.preventDefault();
  const unit=e.deltaMode===1?16:e.deltaMode===2?innerHeight:1;
  pushTravel(clamp(e.deltaY*unit,-240,240)*.0045);
},{passive:false});
$('#timeline').addEventListener('wheel',e=>{e.preventDefault();pushTravel(clamp(e.deltaY,-240,240)*.0045);},{passive:false});
let touchY;
$('#stage').addEventListener('touchstart',e=>{touchY=e.touches[0].clientY;},{passive:true});
$('#stage').addEventListener('touchmove',e=>{if(debug||touchY==null||dialog.open)return;e.preventDefault();const y=e.touches[0].clientY;pushTravel((touchY-y)*.009);touchY=y;},{passive:false});
$('#stage').addEventListener('touchend',()=>{touchY=null;targetTravel=Math.round(targetTravel);});
$('#stage').addEventListener('pointermove',e=>{const r=$('#stage').getBoundingClientRect();pointerX=clamp((e.clientX-r.left)/r.width*2-1,-1,1);pointerY=clamp((e.clientY-r.top)/r.height*2-1,-1,1);dirty=true;});
$('#stage').addEventListener('pointerleave',()=>{pointerX=pointerY=0;dirty=true;});
addEventListener('keydown',e=>{
  if(dialog.open||debug||e.target.closest('select, input, textarea'))return;
  if(['ArrowDown','PageDown','ArrowRight','ArrowUp','PageUp','ArrowLeft','Home','End'].includes(e.key)){
    e.preventDefault();const backwards=['ArrowUp','PageUp','ArrowLeft'].includes(e.key);
    navigateTo(e.key==='Home'?0:e.key==='End'?archive.photos.length-1:Math.round(targetTravel)+(backwards?-1:1));
  }
});

// Enlarged original photo, including the complete left-hand archive behind it.
// Wheel/pinch zooms and pointer drag pans; only the return button is visible.
const dialog=$('#enlarged'),surface=$('#zoom-surface'),zoomImage=$('#zoom-photo');
let zoom=1,panX=0,panY=0,lastFocus,dragged=false;
const pointers=new Map();let pinch;
function paintZoom(){zoomImage.style.transform=`translate(${panX}px,${panY}px) scale(${zoom})`;surface.style.cursor=zoom>1?'grab':'zoom-in';}
function openPhoto(photo){lastFocus=document.activeElement;zoomImage.src=photo.src;zoomImage.alt=photo.description;zoom=1;panX=panY=0;paintZoom();dialog.showModal();document.body.classList.add('modal-open');$('#back').focus();}
function closePhoto(){dialog.close();}
$('#back').addEventListener('click',closePhoto);
dialog.addEventListener('close',()=>{document.body.classList.remove('modal-open');pointers.clear();pinch=null;lastFocus?.focus({preventScroll:true});onScroll();});
surface.addEventListener('wheel',e=>{e.preventDefault();const before=zoom;zoom=clamp(zoom*Math.exp(-e.deltaY*.002),1,8);const x=e.clientX-innerWidth/2,y=e.clientY-innerHeight/2;panX=x-(x-panX)*zoom/before;panY=y-(y-panY)*zoom/before;if(zoom===1)panX=panY=0;paintZoom();},{passive:false});
surface.addEventListener('pointerdown',e=>{surface.setPointerCapture(e.pointerId);pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});dragged=false;surface.classList.add('dragging');if(pointers.size===2){const [a,b]=[...pointers.values()];pinch={distance:Math.hypot(a.x-b.x,a.y-b.y),zoom};}});
surface.addEventListener('pointermove',e=>{const old=pointers.get(e.pointerId);if(!old)return;const dx=e.clientX-old.x,dy=e.clientY-old.y;pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(Math.abs(dx)+Math.abs(dy)>2)dragged=true;if(pointers.size===2&&pinch){const [a,b]=[...pointers.values()];zoom=clamp(pinch.zoom*Math.hypot(a.x-b.x,a.y-b.y)/pinch.distance,1,8);}else if(zoom>1){panX+=dx;panY+=dy;}paintZoom();});
function release(e){pointers.delete(e.pointerId);pinch=null;surface.classList.remove('dragging');}
surface.addEventListener('pointerup',release);surface.addEventListener('pointercancel',release);
surface.addEventListener('click',()=>{if(dragged)return;zoom=zoom===1?2:1;panX=panY=0;paintZoom();});
frame.addEventListener('click',e=>{if(!debug&&current)openPhoto(current);});
frame.tabIndex=0;frame.setAttribute('role','button');frame.setAttribute('aria-label','공간 사진 확대');
frame.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();if(current)openPhoto(current);}});

function render(time=0) {
  if(disposed)return;
  requestAnimationFrame(render);
  const dt=Math.min((time-lastFrame)/1000||1/60,.035);lastFrame=time;
  const displacement=targetTravel-travel;
  travelVelocity+=(displacement*110-travelVelocity*20)*dt;
  travel+=travelVelocity*dt;
  if(Math.abs(displacement)<.0001&&Math.abs(travelVelocity)<.001){travel=targetTravel;travelVelocity=0;}
  const index=clamp(Math.round(travel),0,archive.photos.length-1);
  if(Math.abs(targetTravel-travel)<.6 && current?.id!==archive.photos[index].id)selectPhoto(archive.photos[index]);
  const damping=1-Math.exp(-dt*6);
  entrance+=(1-entrance)*(reduced?1:damping);
  lookX+=(pointerX-lookX)*damping;lookY+=(pointerY-lookY)*damping;
  if(!renderer || !currentTexture || !current || loadingPhoto)return;
  if(!dirty && Math.abs(displacement)<.0001 && entrance>.9999 && Math.abs(pointerX-lookX)+Math.abs(pointerY-lookY)<.0001)return;
  dirty=false;
  camera.aspect=current.width/current.height;
  let yaw=0,pitch=0;
  if(current.spatial&&derived&&currentConfig){
    frameProgress=1-Math.min(1,Math.abs(travel-Math.round(travel))*2);
    const dive=debug?0:entrance;
    yaw=debug||reduced?0:clamp(lookX*THREE.MathUtils.degToRad(15)+travelVelocity*.008,-Math.PI/9,Math.PI/9);
    pitch=debug||reduced?0:clamp(lookY*THREE.MathUtils.degToRad(8),-Math.PI/18,Math.PI/18);
    camera.fov=debug?50:90;
    // Enter through the open front and stop inside the cube, facing the VP wall.
    // Five surfaces acquire independent parallax, without orbiting an outer box.
    const inside=derived.f+derived.side*(.12+.08*frameProgress);
    camera.position.set(derived.cx*dive+lookX*derived.side*.035*dive,
      derived.cy*dive-lookY*derived.side*.025*dive,-inside*dive);
    camera.rotation.set(-pitch,-yaw,0,'YXZ');
    camera.near=derived.side*.005;camera.far=derived.f+derived.side*12;
    camera.updateProjectionMatrix();
    if(debug){camera.projectionMatrix.elements[8]=1-2*currentConfig.vanishingPoint.x;camera.projectionMatrix.elements[9]=2*currentConfig.vanishingPoint.y-1;}
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    for(const mesh of room.children)mesh.material.uniforms.darkening.value=0;
  }else{
    camera.fov=50;camera.position.set(0,0,1/(2*Math.tan(THREE.MathUtils.degToRad(25))));
    camera.rotation.set(0,0,0);camera.near=.01;camera.far=100;camera.updateProjectionMatrix();
  }
  renderer.render(scene,camera);
  window.__signArchive={id:current.id,faces:room.children.map(m=>m.name),depth:derived?.depth??0,
    cubeSide:derived?.side??0,cameraZ:camera.position.z,entrance,travel,targetTravel,
    yawDegrees:THREE.MathUtils.radToDeg(yaw),pitchDegrees:THREE.MathUtils.radToDeg(pitch),
    textureCount:renderer.info.memory.textures,spatialCount:spatial.length};
}

function drawDebug() {
  const svg=$('#debug-overlay');svg.toggleAttribute('hidden',!debug||!derived);if(!debug||!derived)return;
  svg.replaceChildren();
  for(const region of Object.values(derived.regions)) {
    const p=document.createElementNS('http://www.w3.org/2000/svg','polygon');
    p.setAttribute('points',region.source.map(([x,y])=>`${x*1000},${y*1000}`).join(' '));svg.append(p);
  }
  const b=currentConfig.backWall,v=currentConfig.vanishingPoint;
  for(const [name,x,y] of [['vp',v.x,v.y],['tl',b.left,b.top],['br',b.right,b.bottom]]) {
    const dot=document.createElementNS(svg.namespaceURI,'circle');dot.setAttribute('cx',x*1000);dot.setAttribute('cy',y*1000);dot.setAttribute('r',8);dot.dataset.handle=name;svg.append(dot);
  }
}
let debugHandle;
$('#debug-overlay').addEventListener('pointerdown',e=>{debugHandle=e.target.dataset.handle;if(debugHandle)$('#debug-overlay').setPointerCapture(e.pointerId);});
$('#debug-overlay').addEventListener('pointermove',e=>{
  if(!debugHandle)return;
  const r=frame.getBoundingClientRect(),x=clamp((e.clientX-r.left)/r.width,.02,.98),y=clamp((e.clientY-r.top)/r.height,.02,.98);
  const v=currentConfig.vanishingPoint,b=currentConfig.backWall;
  if(debugHandle==='vp'){v.x=clamp(x,b.left+.02,b.right-.02);v.y=clamp(y,b.top+.02,b.bottom-.02);}
  if(debugHandle==='tl'){b.left=Math.min(x,v.x-.02);b.top=Math.min(y,v.y-.02);}
  if(debugHandle==='br'){b.right=Math.max(x,v.x+.02);b.bottom=Math.max(y,v.y+.02);}
  configMap.set(current.id,structuredClone(currentConfig));rebuild();refreshDebugPanel();
});
$('#debug-overlay').addEventListener('pointerup',()=>debugHandle=null);
$('#debug-overlay').addEventListener('pointercancel',()=>debugHandle=null);
function refreshDebugPanel(){if(!debug)return;$('#debug-panel').hidden=!currentConfig;if(!currentConfig)return;$('#debug-json').textContent=JSON.stringify(currentConfig,null,2);$('#sky-mode').value=currentConfig.sky.mode;}
if(debug) {
  const panel=document.createElement('div');panel.id='debug-panel';
  panel.innerHTML='<label>하늘 <select id="sky-mode"><option>photo</option><option>gradient</option><option>far-plane</option></select></label> <button id="export-config">JSON 저장</button><pre id="debug-json"></pre>';
  document.body.append(panel);
  $('#sky-mode').addEventListener('change',e=>{currentConfig.sky.mode=e.target.value;configMap.set(current.id,structuredClone(currentConfig));rebuild();refreshDebugPanel();});
  $('#export-config').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify([...configMap.values()],null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='projections.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
}
// Both spatial photographs and standalone signs participate in navigation.
if(archive.photos.length)await selectPhoto(archive.photos[0]);
await Promise.all([...$('#archive').querySelectorAll('img')].filter(i=>i.loading!=='lazy').map(i=>i.decode().catch(()=>{})));
render();
addEventListener('pagehide',()=>{disposed=true;currentTexture?.dispose();renderer?.dispose();},{once:true});
