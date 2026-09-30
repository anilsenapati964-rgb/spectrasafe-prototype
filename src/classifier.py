"""Deterministic demo classifier, trained only on procedurally generated examples."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def demo_training_data(seed=31):
    rng=np.random.default_rng(seed); X=[]; y=[]
    # Intentionally separated synthetic feature distributions for a stable UI demo.
    for label in [0,1]:
        center=np.r_[np.full(12,.54 if label==0 else .34), np.full(12,.09 if label==0 else .21),
                     np.full(11,1.01 if label==0 else .82), np.full(11,.005 if label==0 else -.09),
                     [.12 if label==0 else .34,.10 if label==0 else .29,.72]]
        for _ in range(100): X.append(center+rng.normal(0,.035, len(center))); y.append(label)
    return np.asarray(X),np.asarray(y)

def get_model(path="models/classifier.pkl"):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        try:
            payload=joblib.load(path)
            if isinstance(payload,dict) and "model" in payload: return payload
        except Exception: pass
    X,y=demo_training_data(); a,b,c,d=train_test_split(X,y,test_size=.25,random_state=7,stratify=y)
    model=RandomForestClassifier(n_estimators=90,max_depth=8,random_state=7,class_weight="balanced").fit(a,c)
    metrics={"accuracy":accuracy_score(d,model.predict(b)),"precision":precision_score(d,model.predict(b),zero_division=0),"recall":recall_score(d,model.predict(b),zero_division=0)}
    payload={"model":model,"metrics":metrics,"source":"synthetic demo data"}
    try: joblib.dump(payload,path)
    except OSError: pass
    return payload
