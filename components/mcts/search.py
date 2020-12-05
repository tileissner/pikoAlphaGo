class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        # Actual MCTS Simulations (E
        for _ in range(0, simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            #TODO reward sollte von NN kommen
            #rewards = getEstimatedRewardFromNode(v)
            v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        # Go DOES NOT expand until terminal node
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
