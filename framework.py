import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
from copy import  deepcopy


class Data:
	""" Data for 2d packing problems. The data is read from a single input file.
	"""

	def __init__(self, input_file):
		self.input_file = input_file
		self.n_containers = 0
		self.containers = []
		self.n_boxes = 0
		self.boxes = []
		self.max_weight = 0
		self.read_data(input_file)

	def read_data(self, input_file):
		""" Reads the container / box dimensions and box weights from a .txt file

		:param input_file:
		"""

		with open(input_file, 'r') as f_in:

			for index, line in enumerate(f_in.read().splitlines()):

				if index == 0:
					self.n_containers = int(line.strip())

				elif 1 <= index <= self.n_containers:
					info = tuple(map(int, line.split()))
					assert len(info) == 2, "Line {} error".format(index)
					self.containers.append(Container(index-1, x_length=info[0], y_length=info[1]))

				elif index == self.n_containers+1:
					self.n_boxes = int(line.strip())

				elif self.n_containers + 1 < index <= self.n_containers + 1 + self.n_boxes:
					info = tuple(map(int, line.split()))
					assert len(info) == 3, "Line {} error".format(index)
					weight = info[2]
					self.boxes.append(Box(index-self.n_containers-2, info[0], info[1], weight))
					if weight >= self.max_weight:
						self.max_weight = weight

				else:
					raise ValueError("Too many lines")


class Box:
	""" A box by definition has an x_length, y_length and a weight. We also allocate each box a specific id. We have also given each box a number of other
	parameters that might be helpful. 

	:param packed: is the box currently packed into a container
	:param container_id: the container id that the box is packed into (-1 if not packed)
	:param x_min: the x-value of the bottom-left corner of the box
	:param x_delta: the length of the box in the x-direction (this depends on the orientation of the box)
	:param y_min: the y-value of the bottom-left corner of the box
	:param y_delta: the length of the box in the y-direction
	:param is_horizontal: a boolean value that indicates if the box is packed horizontally of vertically
	"""

	def __init__(self, id=-1, x_length=0, y_length=0, weight=0):
		self.id = id
		self.x_length = x_length
		self.y_length = y_length
		self.weight = weight
		self.packed = False
		self.container_id = -1
		self.x_min = 0
		self.x_delta = 0
		self.y_min = 0
		self.y_delta = 0
		self.is_horizontal = True

	def pack(self, container_id, x_pos, y_pos, is_horizontal):
		"""A method for packing a box into a container in a specific direction and position"""

		self.container_id = container_id
		self.x_min = x_pos
		self.y_min = y_pos
		self.x_delta = self.x_length if is_horizontal else self.y_length
		self.y_delta = self.y_length if is_horizontal else self.x_length
		self.is_horizontal = is_horizontal
		self.packed = True

	def unpack(self):
		"""A method for unpacking a box from a container"""
		self.packed = False
		self.container_id = -1
		self.x_min = 0
		self.x_delta = 0
		self.y_min = 0
		self.y_delta = 0
		self.is_horizontal = True


class Container:
	"""A container into which the boxes are placed"""

	def __init__(self, id, x_length, y_length):
		"""
		:param id: the id of the container
		:param x_length: the length of the container in the x direction
		:param y_length: the length of the container in the y direction
		:param corners: a list of the current corners of the container. A new container has a single corner in the bottom-left corner.
		:param boxes: the boxes that are contained in the container.
		"""
		self.id = id
		self.x_length = x_length
		self.y_length = y_length
		self.corners = [Corner(x=0, y=0)]
		self.boxes = []

	def unpack(self):
		"""Resets a container to be empty"""
		self.corners = [Corner(x=0, y=0)]
		for box in self.boxes:
			box.unpack()
		self.boxes = []


class Corner:
	""" A corner of a container that a box can be inserted into"""

	def __init__(self, x, y):
		"""
		:param x: the x position of the corner
		:param y: the y position of the corner
		"""
		self.x = x
		self.y = y


class SolutionState:
	""" The state of parameters at a solution to the problem."""


	def __init__(self, containers, boxes, corner_preferences="", direction_preferences=""):
		""" 
		:param containers: list of container objects, where the order the containers appear in the list represent the order
					   		in which they are considered by the heuristic
		:param boxes: list of box objects, where the order the boxes appear in the list represent the order in which they are
					  considered by the heuristic
		:param corner_preferences: list of integers the same length as the list of boxes. The k-th integer in the list corresponds 
					  to the k-th box in boxes. Can be used to change the order that a box considers corners by the heuristic. For
					  example, if we let $i$ represent the integer corner preference and $n$ the number of corners
					  of a container that the box has not been inserted into, we can select the corner at position i mod n in the list 
					  of corners.
	 	:param corner_preferences: list of boolean value the same length as the list of boxes. The k-th boolean in the list corresponds 
					  to the k-th box in boxes. Can be used to change the direction that a box is inserted into the heuristic. For
					  example, if the corner preference is True we will insert the box horitontally, and if it is False we will first try to
					  insert the box vertically into a corner.
		"""

		self.containers = containers
		self.boxes = boxes
		self.corner_preferences = corner_preferences
		self.direction_preferences = direction_preferences

	def objective(self):
		"""
		The objective function is simply the sum of all the box weights that are currently in a container
		"""

		return sum(box.weight for container in self.containers for box in container.boxes)

	def plot(self):
		""" Plots the current solution state. 

		TODO (if you would like to help :-) ): The relative sizes of the boxes and containers can at times be a little buggy. If anyone knows a neater
		way to do this please update and let me know so I can share with others.
		"""

		n_cols = 1000
		n_rows = 2
		n_containers = len(self.containers)
		fig = plt.figure(figsize=(8*n_containers, 10))
		norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
		cmap = cm.plasma
		m = cm.ScalarMappable(norm=norm, cmap=cmap)


		#################
		# Plot Containers
		#################

		g = gridspec.GridSpec(n_rows, n_cols, height_ratios=[3, 1])
		for container in self.containers:
			col_start = int((container.id * n_cols) / float(n_containers))
			col_end = int((container.id + 0.9) * n_cols / float(n_containers))
			ax = fig.add_subplot(g[0, col_start:col_end])

			for box in container.boxes:
				ax.add_patch(matplotlib.patches.Rectangle(
					(box.x_min, box.y_min),
					box.x_delta,
					box.y_delta,
					edgecolor="black",
					facecolor=m.to_rgba(box.weight)))

				plt.text(box.x_min + 0.5 * box.x_delta, box.y_min + 0.5 * box.y_delta, str(box.id), ha="center")

			for corner in container.corners:
				ax.add_patch(matplotlib.patches.Arrow(
					x=corner.x,
					y=corner.y,
					dx=container.x_length/25,
					dy=0,
					width=1
				))
				ax.add_patch(matplotlib.patches.Arrow(
					x=corner.x,
					y=corner.y,
					dx=0,
					dy=container.y_length/25,
					width=1
				))

			plt.xlim([0, container.x_length])
			plt.ylim([0, container.y_length])

		#################
		# Plot Unpacked Boxes
		#################
		unpacked_boxes = [box for box in set(self.boxes)-set(box for container in self.containers for box in container.boxes)]
		ax = fig.add_subplot(g[1, 0:n_cols])
		ax.axis('off')
		x_pos = 0
		max_y = 1
		if unpacked_boxes:
			for index, box in enumerate(unpacked_boxes):
				ax.add_patch(matplotlib.patches.Rectangle(
					(x_pos, 0),
					box.x_length,
					box.y_length,
					edgecolor="black",
					facecolor=m.to_rgba(box.weight)))

				plt.text(x_pos + 0.5 * box.x_length, 0.5 * box.y_length, str(box.id), ha="center")

				x_pos += box.x_length + 1
				max_y = max(max_y, box.y_length)

			plt.xlim([0, x_pos])
			plt.ylim([0, max_y * 2])


		#plt.colorbar(mappable=m)

		plt.show()


