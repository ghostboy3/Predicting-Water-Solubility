import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('curated-solubility-dataset.csv')
# print(data.head())

# convert smiles to rdkit molecules
data['Molecule'] = data['SMILES'].apply(Chem.MolFromSmiles)

# compute molecular descriptors
data['LogP'] = data['Molecule'].apply(Descriptors.MolLogP)
data['MolecularWeight'] = data['Molecule'].apply(Descriptors.MolWt)

X = data[['LogP', 'MolecularWeight']]
y = data['Solubility']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

val_score = model.score(X_val, y_val)
print(f"Validation R^2 score: {val_score:.3f}")

new_smiles = '[Zn++].CC(c1ccccc1)c2cc(C(C)c3ccccc3)c(O)c(c2)C([O-])=O.CC(c4ccccc4)c5cc(C(C)c6ccccc6)c(O)c(c5)C([O-])=O'#'COC(=O)c1c[nH]c2cc(OC(C)C)c(OC(C)C)cc2c1=O'
new_molecule = Chem.MolFromSmiles(new_smiles)
new_logp = Descriptors.MolLogP(new_molecule)
print("logp: " + str(new_logp))
new_mw = Descriptors.MolWt(new_molecule)
print("new_mw: " + str(new_mw))

predicted_solubility = model.predict([[new_logp, new_mw]])
print(f"Predicted solubility: {predicted_solubility[0]:.2f} mg/L")
