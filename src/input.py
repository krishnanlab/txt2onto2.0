import pandas as pd
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-onto",
                        help="Ontology term to build a model for",
                        required=True,
                        type=str)
    parser.add_argument("-gs",
                        help="Ontology term to build a model for",
                        required=True,
                        type=str)
    parser.add_argument("-text",
                        help="Ontology term to build a model for",
                        required=True,
                        type=str)
    parser.add_argument("-id",
                        help="Ontology term to build a model for",
                        required=True,
                        type=str)
    parser.add_argument("-out",
                        help="Directory to save trained model to",
                        required=True,
                        type=str)
    args = parser.parse_args()

    # load gs
    gs_df = pd.read_csv(args.gs, header=0, index_col=0)
    if args.onto not in list(gs_df.columns):
        raise Exception(f'Provided ontology term {args.onto} does not exist')

    # load id
    ID_list = list(pd.read_csv(args.id, header=None, index_col=None, sep='\t')[0])
    if len(ID_list) != len(set(ID_list)):
        raise Exception(f'ID provided in {args.id} is not unique\n')

    # load text
    desc_df = pd.read_csv(args.text, header=None, index_col=None, sep='\t')
    desc_df.columns = ['text']
    desc_df['ID'] = ID_list
    desc_df['label'] = list(gs_df.loc[desc_df['ID'], args.onto])
    desc_df = desc_df[desc_df['label'] != 0]
    desc_df['label'] = [0 if i == -1 else 1 for i in desc_df['label']]
    desc_df = desc_df.loc[:, ['ID', 'label', 'text']]

    # save output
    desc_df.to_csv(
        '%s/%s__train_input.tsv' % (args.out, args.onto.replace(':', '_')),
        header=True,
        index=False,
        sep='\t',
    )
