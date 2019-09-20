from Framework.PlayerController import PlayerController

class PlayerNeatController(PlayerController):
	def control(self, inputs, controller):
		activator = controller.activate(inputs)

		# takes decisions about sprite actions
		if activator[0] > 0.5:
			left = 1
		else:
			left = 0

		if activator[1] > 0.5:
			right = 1
		else:
			right = 0

		if activator[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if activator[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if activator[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]