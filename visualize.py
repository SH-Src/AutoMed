from models_new import genotypes

import os
os.environ["PATH"] += os.pathsep + 'D://Graphviz/bin/'
from models_new.genotypes import Genotype
from graphviz import Digraph


def plot(genotypes, filename):

    if genotypes == None:
        return

    g = Digraph(
            format='pdf',
            edge_attr=dict(fontname="helvetica", penwidth='1.0', fontsize='40'),
            node_attr=dict(style='rounded, filled', shape='rect', align='center',
                           height='0.5', width='0.5', penwidth='1.5',
                           fontname="helvetica", fontsize='40'),
            engine='dot', filename=filename)
    g.attr(rankdir='LR')

    with g.subgraph(name='cluster_0') as c:
        c.node_attr.update(style='rounded, filled', color='lightblue')
        for s in range(2):
            for op, i in genotypes.time[s:s+s+1]:
                c.edge('x_T_' + str(i), 'x_T_' + str(s+1), label=op)
        c.attr(label='Time Cell', style='rounded', color='orange', fontsize='40')

    with g.subgraph(name='cluster_1') as c:
        c.attr(label='Diagnosis Cell', style='rounded', color='orange', fontsize='40')
        c.node_attr.update(style='rounded, filled', color='lightgreen')
        for s in range(2):
            for op, i in genotypes.ehr[s:s + s + 1]:
                c.edge('x_D_' + str(i), 'x_D_' + str(s + 1), label=op)


    with g.subgraph(name='cluster_2') as c:
        c.attr(label='Fusion Cell', style='rounded', color='orange', fontsize='40')
        c.node_attr.update(style='rounded, filled', color='lightyellow')
        for s in range(2):
            for op, i in genotypes.fuse[s:s + s + 1]:
                c.edge('x_F_' + str(i), 'x_F_' + str(s + 1), label=op)


    e, t = genotypes.select[0], genotypes.select[1]
    g.node('CatFC', color='palegoldenrod', style='rounded, filled')
    g.edge('x_D_'+ str(e), 'CatFC', color='red')
    g.edge('x_T_'+ str(t), 'CatFC', color='red')
    g.edge('CatFC', 'x_F_0', color='blue')
    g.engine = 'dot'
    g.view()

if __name__ == '__main__':
    genotypes = Genotype(time=[('conv', 0), ('conv', 0), ('conv', 1)],
                         ehr=[('ffn', 0), ('identity', 0), ('identity', 1)],
                         fuse=[('attention', 0), ('rnn', 0), ('identity', 1)], select=[1, 2])

    plot(genotypes, 'hf.gv')

    genotypes = Genotype(time=[('conv', 0), ('conv', 0), ('conv', 1)],
                         ehr=[('zero', 0), ('zero', 0), ('zero', 1)],
                         fuse=[('attention', 0), ('attention', 0), ('ffn', 1)], select=[0, 2])
    plot(genotypes, 'copd.gv')

    genotypes = Genotype(time=[('conv', 0), ('attention', 0), ('conv', 1)],
                         ehr=[('attention', 0), ('identity', 0), ('identity', 1)],
                         fuse=[('zero', 0), ('rnn', 0), ('zero', 1)], select=[1, 2])
    plot(genotypes, 'amnesia.gv')

    genotypes = Genotype(time=[('conv', 0), ('conv', 0), ('conv', 1)], ehr=[('identity', 0), ('zero', 0), ('zero', 1)], fuse=[('conv', 0), ('rnn', 0), ('zero', 1)], select=[0, 2])
    plot(genotypes, 'kidney.gv')

    genotypes = Genotype(time=[('conv', 0), ('conv', 0), ('conv', 1)], ehr=[('conv', 0), ('conv', 0), ('attention', 1)], fuse=[('ffn', 0), ('attention', 0), ('attention', 1)], select=[1, 2])
    plot(genotypes, 'dementia.gv')


