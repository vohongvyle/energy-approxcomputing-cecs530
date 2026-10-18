import json, gzip, base64, numpy as np, os
from run_experiment import load_data, train, QuantMLP
Xtr,Xte,ytr,yte = load_data("digits",0)
clf = train(Xtr,ytr,64,0)
q = QuantMLP(clf,Xtr)
circ = json.load(open("luts/circuits.json"))
names = sorted(circ, key=lambda n:-circ[n]["pwr"])
luts = np.stack([np.load(f"luts/{n}.npy") for n in names])
print("lut max", luts.max(), luts.min())
blob = base64.b64encode(gzip.compress(luts.astype("<u2").tobytes(),9)).decode()
print("lut b64 MB", len(blob)/1e6)
d = dict(
 w1=(q.sg1*q.m1).astype(int).T.flatten().tolist(),  # [out][in]
 w2=(q.sg2*q.m2).astype(int).T.flatten().tolist(),
 b1=np.round(q.b1,6).tolist(), b2=np.round(q.b2,6).tolist(),
 sx=q.sx, sw1=q.sw1, sh=q.sh, sw2=q.sw2, nin=64, nh=64, nout=10,
 X=np.round(Xte*16).astype(int).flatten().tolist(), y=yte.tolist(),
 circuits=[dict(name=n, pwr=circ[n]["pwr"], mae=circ[n]["mae%"], wce=circ[n]["wce%"],
                area=circ[n]["area"], bias=round(circ[n]["measured_bias"],1)) for n in names],
 float_acc=float((clf.predict(Xte)==yte).mean()))
open("web_data.js","w").write("window.DATA="+json.dumps(d,separators=(",",":"))+";\nwindow.LUTS_GZ='"+blob+"';\n")
print(os.path.getsize("web_data.js")/1e6, "MB")
