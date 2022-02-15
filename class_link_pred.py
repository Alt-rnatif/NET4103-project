from abc import ABC
#from abc import abstractmethod
from math import log
#import progressbar


class LinkPrediction(ABC):
    def __init__( self, graph ):
        """
        Constructor

        Parameters
        ----------
        graph : Networkx graph
        """
        self.graph = graph
        self.N = len(graph)

    def neighbors( self, v ):
        """
        Return the neighbors list of a node

        Parameters
        ----------
        v : int
        node id

        Return
        ------
        neighbors_list : python list
        """
        neighbors_list = self.graph.neighbors(v)
        return list(neighbors_list)

    '''
    @abstractmethod
    def fit( self ):
        raise NotImplementedError(" Fit must be implemented ")
    '''


class CommonNeighbors(LinkPrediction):

    def __init__( self, graph ):
        super(CommonNeighbors, self).__init__(graph)

    def evaluate( self, u, v ):
        assert u in self.graph
        assert v in self.graph
        neigh_u = self.neighbors(u)
        neigh_v = self.neighbors(v)
        score = 0
        for i in neigh_u:
            if i in neigh_v:
                score += 1
        return score


class Jaccard(LinkPrediction):

    def __init__( self, graph ):
        super(Jaccard, self).__init__(graph)

    def evaluate( self, u, v ):
        assert u in self.graph
        assert v in self.graph
        neigh_u = self.neighbors(u)
        neigh_v = self.neighbors(v)
        intersection = 0
        for i in neigh_u:
            if i in neigh_v:
                intersection += 1
        union = list(dict.fromkeys(neigh_u + neigh_v))
        try:
            return intersection / len(union)
        except:
            return 0


class AdamicAdar(LinkPrediction):

    def __init__( self, graph ):
        super(AdamicAdar, self).__init__(graph)

    def evaluate( self, u, v ):
        assert u in self.graph
        assert v in self.graph
        neigh_u = self.neighbors(u)
        neigh_v = self.neighbors(v)
        intersection = []
        for i in neigh_u:
            if i in neigh_v:
                intersection.append(i)
        score = 0
        for i in intersection:
            score += 1 / log(len(self.neighbors(i)))
        return score
