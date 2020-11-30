from pygeo.intersect import (
    intersect,
    _intersect_ray_with_sphere,
    _intersect_ray_with_triangle,
)
from pygeo.objects import Ray, Sphere, Triangle, Point, Vector
import numpy as np

# intersect


# _intersect_ray_with_sphere
def test_intersect_ray_and_sphere_with_same_origin():
    assert(_intersect_ray_with_sphere(Ray(Point((0, 0, 0)),Vector((1,1,1))), Sphere(Point((0, 0, 0)), 2)) == Point([1.15, 1.15, 1.15])) is True

def test_intersect_ray_and_sphere_with_different_origin():
    assert(_intersect_ray_with_sphere(Ray(Point((0, 0, 0)),Vector((1,1,1))), Sphere(Point((3, 3, 3)), 1)) == (Point([2.42, 2.42, 2.42]), Point([3.58, 3.58, 3.58]))) is True

def test_intersect_ray_and_sphere_with_different_origin_and_opposite_direction():
    assert(_intersect_ray_with_sphere(Ray(Point((0, 0, 0)),Vector((-1,-1,-1))), Sphere(Point((3, 3, 3)), 1)) == "No Intersecction") is True

def test_intersect_ray_and_sphere_with_different_origin_and_no_intersection():
    assert(_intersect_ray_with_sphere(Ray(Point((0, 0, 0)),Vector((-1,-1,-1))), Sphere(Point((4, 4, 0)), 1)) == "No Intersecction") is True


# _intersect_ray_with_triangle
