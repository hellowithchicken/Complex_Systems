def check_for_neighbours(row, col, grid, direction):
  """
  Looks in a specified direction if there is a neighbour
  """
  if direction == "up" and (row - 1 > -1):
    return grid[row - 1, col] == 1
  if direction == "down" and (row + 1 < len(grid)):
    return grid[row + 1, col] == 1
  if direction == "right" and (col + 1 < len(grid)):
    return grid[row, col + 1] == 1
  if direction == "left" and (col - 1 > -1):
    return grid[row, col - 1] == 1
  return False

def get_network(grid, simplify = False):
  """
  Takes a grid of 1s and 0s and converts it into a network.
  Simply - removed nodes of degree 2.
  """
  # initialise network
  G=nx.Graph()

  node_number = 0
  nodes = []
  links = []
  pos = {}
  for row in range(len(grid)):
    for col in range(len(grid)):
      # update node number
      node_number += 1
      # check if there is anything in cell - if there is not, skip it completely to save time
      if grid[row][col] == 0:
        continue
      # add node to the list
      nodes.append(node_number)
      # get coordinates of the node
      coordinates  = (col, row * -1)
      pos[node_number] = coordinates
      # check for neighbors and if the neighbor exists add a link
      # check up
      if check_for_neighbours(row, col, grid, "up"):
        links.append((node_number, node_number - len(grid)))
      # check down
      if check_for_neighbours(row, col, grid, "down"):
        links.append((node_number, node_number + len(grid)))
      # check left
      if check_for_neighbours(row, col, grid, "left"):
        links.append((node_number, node_number - 1))
      # check down
      if check_for_neighbours(row, col, grid, "right"):
        links.append((node_number, node_number + 1))


  # add nodes to the graph
  G.add_nodes_from(nodes)
  # add links to the graph
  G.add_edges_from(links)

  if simplify:
    for node in list(G.nodes()):
      if G.degree(node) == 2:
          edges = list(G.edges(node))
          G.add_edge(edges[0][1], edges[1][1])
          G.remove_node(node)

  nx.draw(G, pos)