"""
Reference Implementation for bezier_curve_length
This serves as the ground truth for testing LLM-generated implementations.
"""

import numpy as np
import math


def bezier_curve_length(control_points, num_samples):
    """
    Compute the length of a 3D Bezier curve using numerical integration.
    
    The function calculates the arc length of a Bezier curve by integrating
    the magnitude of its derivative using Simpson's composite rule.
    
    Args:
        control_points: List or array of 3D control points, shape (n, 3)
                       where n is the number of control points (degree+1)
        num_samples: Number of subintervals for the Simpson integration
    
    Returns:
        float: The length of the Bezier curve
    """
    def binomial_coefficient(n, i):
        """Compute binomial coefficient C(n, i) = n! / (i! * (n-i)!)"""
        return math.factorial(n) // (math.factorial(i) * math.factorial(n - i))

    def derivative_bezier(t):
        """
        Compute the derivative of the Bezier curve at parameter t.
        
        The derivative of a Bezier curve of degree n with control points P_i is:
        B'(t) = n * sum(i=0 to n-1) C(n-1, i) * (1-t)^(n-1-i) * t^i * (P_(i+1) - P_i)
        """
        n = len(control_points) - 1
        point = np.array([0.0, 0.0, 0.0])
        for i, (p1, p2) in enumerate(zip(control_points[:-1], control_points[1:])):
            point += (
                n
                * binomial_coefficient(n - 1, i)
                * (1 - t) ** (n - 1 - i)
                * t**i
                * (np.array(p2) - np.array(p1))
            )
        return point

    def simpson_integral(a, b, num_samples):
        """
        Compute the integral of ||B'(t)|| from a to b using Simpson's rule.
        
        Simpson's rule: integral ≈ (h/3) * [f(a) + 4*sum(f(x_odd)) + 2*sum(f(x_even)) + f(b)]
        where h = (b-a)/num_samples
        """
        h = (b - a) / num_samples
        lt = derivative_bezier
        sum1 = sum(
            np.linalg.norm(lt(a + i * h))
            for i in range(1, num_samples, 2)
        )
        sum2 = sum(
            np.linalg.norm(lt(a + i * h))
            for i in range(2, num_samples - 1, 2)
        )
        return (
            (
                np.linalg.norm(lt(a))
                + 4 * sum1
                + 2 * sum2
                + np.linalg.norm(lt(b))
            )
            * h
            / 3
        )

    # Compute the length of the 3D Bezier curve using composite Simpson's rule
    length = 0.0
    for i in range(num_samples):
        t0 = i / num_samples
        t1 = (i + 1) / num_samples
        length += simpson_integral(t0, t1, num_samples)

    return length

