import os
import csv
from rdkit.Chem import MolFromSmiles

import inspect
import rdkit
print(inspect.getmodule(rdkit.Chem))

def smi_to_csv(pos_file, neg_file, output_file):
    data = []

    with open(pos_file, "r") as pos_handle:
        for line in pos_handle:
            data.append({"smiles": line.split("\t")[0], "target": 1})
    with open(neg_file, "r") as neg_handle:
        for line in neg_handle:
            data.append({"smiles": line.split("\t")[0], "target": 0})

    header = ["smiles", "target"]

    with open(output_file, "w+") as output_handle:
        csv_writer = csv.DictWriter(output_handle, header, "")
        csv_writer.writeheader()
        csv_writer.writerows(data)

def make_files(folder=[], filenames=[]):
    if folder:
        path = folder[0]
        for path_folder in folder[1:]:
            path = "%s%s%s" % (path, os.sep, path_folder)
            try:
                os.makedirs(path)
            except:
                pass
        filenames = ["%s%s%s" % (path, os.sep, filename) for filename in filenames]

    for filename in filenames:
        open(filename, "a+").close()
    
    return filenames

def remove_unkekulizable(csv_file):
    data = []
    headers = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        headers = next(iter(reader))
        file.seek(0)
        reader = list(csv.DictReader(file))
        len_with_kek = len(reader)
        pos_kek_num = 0
        for line in reader:
            if MolFromSmiles(line["smiles"]) != None:
                data.append(line)
            else:
                if line["target"] == 1:
                    pos_kek_num+=1
        len_no_kek = len(data)

        print("removed %s unkekable mols out of %s = %s%% from %s. %s were positive" % (len_with_kek-len_no_kek, 
            len_with_kek, float(len_with_kek-len_no_kek)/len_with_kek, csv_file, pos_kek_num))

    with open(csv_file, "w+") as file:
        writer = csv.DictWriter(file, headers, "")
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def parse_crk3d_file(filename):
    molecules = []
    with open(filename, "r") as input_file:
        while True:
            try:
                molname = next(input_file).strip()
                cur_mol = Mol(molname)
            except StopIteration:
                break

            ligand_decoy = next(input_file).strip()
            if ligand_decoy == "decoy":
                cur_mol.ligand = False
            elif ligand_decoy == "ligand":
                cur_mol.ligand = True
            else:
                raise Exception("not ligand or decoy!!!!")

            cur_mol.smile = next(input_file).strip()

            assert("<Property Type=\"ModelStructure\">" == next(input_file).strip())
            assert("<Structure3D>" == next(input_file).strip())

            charge_spin = re.match(r"<Group Charge=\"([0-9]+)\" Spin=\"([0-9])+\">", next(input_file).strip())
            charge = charge_spin.group(1)
            cur_mol.charge = charge
            spin = charge_spin.group(2)
            cur_mol.spin = spin

            while True:
                first_line = next(input_file).strip()
                if re.match(r"<Atom.*", first_line):
                    atom_id = int(re.match(r"<Atom ID=\"([0-9]*)\">", first_line).group(1)) - 1
                    X = float(re.match(r"<X>([0-9\.\-]+)</X>", next(input_file).strip()).group(1))
                    Y = float(re.match(r"<Y>([0-9\.\-]+)</Y>", next(input_file).strip()).group(1))
                    Z = float(re.match(r"<Z>([0-9\.\-]+)</Z>", next(input_file).strip()).group(1))
                    element = re.match(r"<Element>([a-zA-Z]+)</Element>", next(input_file).strip()).group(1)
                    assert(next(input_file).strip() == "</Atom>")

                    cur_mol.atoms.append(Atom(element, [X, Y, Z], atom_id, cur_mol))

                elif first_line == "<Bond>":
                    atom_1 = int(re.match(r"<From>([0-9\.\-]+)</From>", next(input_file).strip()).group(1)) - 1
                    atom_2 = int(re.match(r"<To>([0-9\.\-]+)</To>", next(input_file).strip()).group(1)) - 1
                    order = float(re.match(r"<Order>([0-9\.\-]+)</Order>", next(input_file).strip()).group(1))
                    style = float(re.match(r"<Style>([0-9\.\-]+)</Style>", next(input_file).strip()).group(1))
                    assert(next(input_file).strip() == "</Bond>")

                    new_bond = Bond(atom_1, atom_2, order, style, cur_mol)
                    cur_mol.bonds[atom_1][atom_2].append(new_bond)
                    # cur_mol.bonds[atom_2][atom_1].append(new_bond)
                    cur_mol.bond_list.append(new_bond)

                elif first_line == "</Group>":
                    assert("</Structure3D>" == next(input_file).strip())
                    assert("</Property>" == next(input_file).strip())

                    break
                else:
                    raise Exception()

            molecules.append(cur_mol)

    return molecules

def raw_to_csv_files(raw_files, csv_file):
    active_file, background_file = raw_files
    smi_to_csv(active_file, background_file, csv_file)
    remove_unkekulizable(csv_file)
