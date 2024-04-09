import pygame
import math

class Node:
    def __init__(self, number, layer):
        # Each node has a unique number associated with it
        self.number = number
        # defines to which layer does the node belong
        self.layer = layer
        # for weighted sum of all the incoming genes(connections)
        self.sum = 0 
        # for output =sigmoid(sum)
        self.output=0 
        # for storing certain value required during backpropogation
        self.value=0 
        # list of all genes incoming genes to a node
        self.in_genes = []
        # Sigmoid activation
        self.sigmoid = lambda x : 1 / (1 + math.exp(-x))

        # For Showing
        self.color = (255, 255, 255)
        self.bcolor = (0, 0, 0)
        self.radius = 7
        self.border_radius = 3
        self.pos = [0, 0]
        pass

    # returns a clone of the node
    def clone(self):
        n = Node(self.number, self.layer)
        n.output = self.output
        n.pos = self.pos
        return n

    # Calculates the sum and output at particular node
    def calculate(self):
        if self.layer == 0:  # layer = 0 -> input layer
            print('No calculations for first layer')
            return

        s = 0
        for g in self.in_genes:  # in_genes contains all the incoming genes
            if g.enabled:  #  genes can be enabled/disabled while mutating
                # weighted sum 
                s += (g.in_node.output * g.weight)
        self.sum = s
        # applying activation function to weighted sum to calucate its output,and rounding of to 4 decimal places
        self.output = round(self.sigmoid(self.sum),4)
        pass

    # Only for showing
    def show(self, ds):
        pygame.draw.circle(ds, self.bcolor, self.pos, self.radius + self.border_radius)
        pygame.draw.circle(ds, self.color, self.pos, self.radius)
        pass