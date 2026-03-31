"""
Reference Implementation for get_densify_rate
This serves as the ground truth for testing LLM-generated implementations.
"""


class TrainingScheduler():
	"""
	DashGaussian training scheduler of resolution and primitive number.
	"""
	
	def get_densify_rate(self, iteration, cur_n_gaussian, cur_scale=None):
		"""
		Calculate the densification rate for Gaussian primitives.
		
		Args:
			iteration: Current training iteration
			cur_n_gaussian: Current number of Gaussians
			cur_scale: Current resolution scale (required when densify_mode is "freq")
		
		Returns:
			The densification rate as a float value
		"""
		if self.densify_mode == "free":
			return 1.0
		elif self.densify_mode == "freq":
			assert cur_scale is not None
			if self.densification_interval + iteration < self.increase_reso_until:
				next_n_gaussian = int((self.max_n_gaussian - self.init_n_gaussian) / cur_scale**(2 - iteration / self.densify_until_iter)) + self.init_n_gaussian
			else:
				next_n_gaussian = self.max_n_gaussian
			return min(max((next_n_gaussian - cur_n_gaussian) / cur_n_gaussian, 0.), self.max_densify_rate_per_step)
		else:
			raise NotImplementedError("Densify mode '{}' is not implemented.".format(self.densify_mode))

