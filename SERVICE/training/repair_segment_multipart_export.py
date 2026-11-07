"""Export repair: retain multipart identity and stitch only unambiguous touching core continuations."""
import os,sys,shutil,json,collections,hashlib
from pathlib import Path
import cv2,numpy as np
from ultralytics.data.converter import merge_multi_segment
from segment_supplementary_core import read_json,write_json,sha
ROOT=Path.cwd();OLD=ROOT/'.work/segmentation_supplementary_20261023/training';NEW=ROOT/'.work/segmentation_supplementary_20261023/training_v2'

def stitch(masks,width):
 parent=list(range(len(masks)));events=[];ambiguous=[]
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for seam in [k*(width//6) for k in range(1,6)]:
  left=[i for i,m in enumerate(masks) if m[:,seam-1].any() and not m[:,seam:].any()];right=[i for i,m in enumerate(masks) if m[:,seam].any() and not m[:,:seam].any()]
  contacts=[]
  for a in left:
   yy=masks[a][:,seam-1].astype(np.uint8);near=cv2.dilate(yy[:,None],np.ones((3,1),np.uint8))[:,0]>0
   for b in right:
    if (near&masks[b][:,seam]).any():contacts.append((a,b))
  ca=collections.Counter(a for a,b in contacts);cb=collections.Counter(b for a,b in contacts)
  for a,b in contacts:
   if ca[a]==cb[b]==1:parent[find(b)]=find(a);events.append({'seam':seam,'left':a,'right':b})
   else:ambiguous.append({'seam':seam,'left':a,'right':b})
 groups=collections.defaultdict(list)
 for i in range(len(masks)):groups[find(i)].append(i)
 result=[np.maximum.reduce([masks[i] for i in ids]) for ids in groups.values()]
 old=np.maximum.reduce(masks) if masks else np.zeros((1,width),bool);new=np.maximum.reduce(result) if result else old
 assert np.array_equal(old,new)
 return result,list(groups.values()),events,ambiguous

def contour_path(mask):
 cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);parts=[]
 for c in cs:
  if len(c)<3 or cv2.contourArea(c)<=0:
   component=np.zeros_like(mask,np.uint8);cv2.drawContours(component,[c],-1,1,1);up=cv2.resize(component,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST);q=cv2.findContours(up,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1,2);points=(q//8).astype(float)+.1+.3*(q%8)/7
  else:points=c.reshape(-1,2).astype(float)+.25
  parts.append(points)
 if len(parts)==1:return parts[0]
 return np.concatenate(merge_multi_segment([p.ravel().tolist() for p in parts]))

def main():
 resume='--resume' in sys.argv
 if NEW.exists() and not resume:raise FileExistsError(NEW)
 (NEW/'input').mkdir(parents=True,exist_ok=resume);shutil.copytree(OLD/'masks',NEW/'masks',dirs_exist_ok=resume);shutil.copytree(OLD/'crops/images',NEW/'crops/images',dirs_exist_ok=resume);(NEW/'crops/labels').mkdir(exist_ok=resume);(NEW/'crops/targets').mkdir(exist_ok=resume);(NEW/'crops/previews').mkdir(exist_ok=resume)
 for name in ['items.json','annotations.json','export_receipt.json']:shutil.copy2(OLD/'input'/name,NEW/'input'/name)
 items=read_json(NEW/'input/items.json');anns=read_json(NEW/'input/annotations.json');oldrows=read_json(OLD/'crops/manifest.json');oldbyid={r['id']:r for r in oldrows};records=[];audit=[]
 for item in items['items']:
  originals=[np.squeeze(cv2.imread(str(f),0))>0 for f in sorted((NEW/'masks'/item['id']).glob('[0-9]*.png'))];masks,groups,events,ambiguous=stitch(originals,item['width']);audit.append({'id':item['id'],'original_instances':len(originals),'stitched_instances':len(masks),'multipart_identity_retained':True,'stitches':events,'ambiguous_contacts_not_merged':ambiguous,'union_pixels_exact':True,'groups':[[anns['bands'][item['id']]['instances'][i]['id'] for i in ids] for ids in groups]})
  for ti,(a,b) in enumerate(item['tile_positions']):
   identifier=f'{item["id"]}_tile{ti}';old=oldbyid[identifier];h,w=old['shape_hw'];targets=[];parentids=[];lines=[]
   for mask,ids in zip(masks,groups):
    tile=mask[:,a:b]
    if not tile.any():continue
    targets.append(tile.astype(np.uint8));parentids.append([anns['bands'][item['id']]['instances'][i]['id'] for i in ids]);points=contour_path(tile);points/=np.asarray([w,h]);lines.append('0 '+' '.join(f'{v:.8f}' for v in points.ravel()))
   truth=np.maximum.reduce(targets)>0 if targets else np.zeros((h,w),bool)
   with np.load(old['target_masks']) as z:assert np.array_equal(truth,z['review_union'].astype(bool))
   label=NEW/'crops/labels'/f'{identifier}.txt';label.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8');npz=NEW/'crops/targets'/f'{identifier}.npz';np.savez_compressed(npz,masks=np.asarray(targets,np.uint8) if targets else np.empty((0,h,w),np.uint8),review_union=truth.astype(np.uint8))
   r={**old,'image':str(NEW/'crops/images'/f'{identifier}.png'),'label':str(label),'target_masks':str(npz),'objects':len(targets),'parent_instance_ids':parentids,'label_sha256':sha(label),'instance_policy':'preserve each reviewed multipart mask; stitch only unique touching continuations at original core seam; do not bridge pixels','training_mask_authority':'target_masks NPZ, required ExactYOLODataset; txt paths are bbox proxies only'};records.append(r)
   source=OLD/'crops/previews'/f'{identifier}.jpg';shutil.copy2(source,NEW/'crops/previews'/source.name)
 assert len(records)==324 and sum(r['background'] for r in records)==8
 write_json(NEW/'crops/manifest.json',records);write_json(NEW/'instance_repair_audit.json',audit);receipt=read_json(NEW/'input/export_receipt.json');receipt.update(export_repair='Multipart identity and unique seam stitching with pixel-exact raw loss masks; original 324 crop unions unchanged',invalid_prior_run=str(OLD),training_started=False);write_json(NEW/'input/export_receipt.json',receipt)
 write_json(NEW/'repair_receipt.json',{'prior_run':str(OLD),'prior_run_status':'technical stop after 3 epochs, invalid instance export; never resume or use its metrics as quality result','user_review_annotations_sha256':sha(NEW/'input/annotations.json'),'original_reviewed_pixels_unchanged':True,'all_324_crop_unions_identical_to_prior_canvas_raster':True,'split_must_remain_identical':True,'repairs':['Preserve all disconnected components of one reviewed ID as one instance.','Stitch only one-to-one pixel-touching continuations at known core seams; ambiguous contacts remain separate.','Use raw NPZ instance masks in train and val; txt support paths never define new training mask pixels.','Disable runtime geometric augmentation; original old hard/shifted examples and photometric variants remain.'],'total_source_instances_before':sum(a['original_instances'] for a in audit),'total_source_instances_after':sum(a['stitched_instances'] for a in audit),'new_crop_instances':sum(r['objects'] for r in records)})
 print(json.dumps(read_json(NEW/'repair_receipt.json'),indent=2),flush=True)
if __name__=='__main__':main()
