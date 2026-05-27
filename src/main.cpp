#include <cstring>
#include <filesystem>
#define _CRT_SECURE_NO_WARNINGS 1

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "../include/stb_image_write.h"

#include "../include/lbfgs.h"

#include "../include/nanoflann_utils.h"

#include <omp.h>

#include "../include/nanoflann.hpp"

double sqr(double x) { return x * x; };

const double eps = 1e-10;

class Vector {
  public:
    explicit Vector(double x = 0, double y = 0) {
        data[0] = x;
        data[1] = y;
        data[2] = 0;
    }
    double norm2() const { return data[0] * data[0] + data[1] * data[1]; }
    double norm() const { return sqrt(norm2()); }
    void normalize() {
        double n = norm();
        data[0] /= n;
        data[1] /= n;
    }
    double operator[](int i) const { return data[i]; };
    double &operator[](int i) { return data[i]; };
    double data[3];
};

Vector operator+(const Vector &a, const Vector &b) {
    return Vector(a[0] + b[0], a[1] + b[1]);
}
Vector operator-(const Vector &a, const Vector &b) {
    return Vector(a[0] - b[0], a[1] - b[1]);
}
Vector operator*(const double a, const Vector &b) {
    return Vector(a * b[0], a * b[1]);
}
Vector operator*(const Vector &a, const double b) {
    return Vector(a[0] * b, a[1] * b);
}
Vector operator/(const Vector &a, const double b) {
    return Vector(a[0] / b, a[1] / b);
}
double dot(const Vector &a, const Vector &b) {
    return a[0] * b[0] + a[1] * b[1];
}

class Polygon {
  public:
    double area() {
        if (vertices.size() < 3)
            return 0;
        // Compute the area of the polygon
        double res = 0;
        for (size_t i = 0; i < vertices.size(); i++) {
            const auto &A = vertices[i];
            const auto &B = vertices[(i + 1) % vertices.size()];
            res += (A[0] * B[1] - A[1] * B[0]);
        }
        return 0.5 * std::abs(res);
    }

    Vector centroid() {
        if (vertices.size() < 3)
            return Vector(0, 0);
        // Compute the centroid of the polygon
        Vector result(0, 0);
        double total_signed_area = 0;

        const auto &A = vertices[0];
        for (size_t i = 1; i < vertices.size() - 1; i++) {
            const auto &B = vertices[i];
            const auto &C = vertices[i + 1];
            // Polygon V;
            // V.vertices.emplace_back(A);
            // V.vertices.emplace_back(B);
            // V.vertices.emplace_back(C);
            double tri_area = 0.5 * ((B[0] - A[0]) * (C[1] - A[1]) -
                                     (B[1] - A[1]) * (C[0] - A[0]));
            result = result + (A + B + C) / 3.0 * tri_area;
            total_signed_area += tri_area;
        }

        if (std::abs(total_signed_area) < 1e-12)
            return A;

        return result / total_signed_area;
    }

    double integral_square_distance(const Vector &Pi) {
        if (vertices.size() < 3)
            return 0;

        double res = 0;

        const auto &A = vertices[0];
        // Compute the integral of ||x-Pi||^2 over the polygon
        for (size_t i = 1; i < vertices.size() - 1; i++) {
            const auto &B = vertices[i];
            const auto &C = vertices[i + 1];
            // Polygon V;
            // V.vertices.emplace_back(A);
            // V.vertices.emplace_back(B);
            // V.vertices.emplace_back(C);
            Vector V[3] = {A, B, C};
            double tri_area = 0.5 * std::abs((B[0] - A[0]) * (C[1] - A[1]) -
                                             (B[1] - A[1]) * (C[0] - A[0]));
            double sub_res = 0;
            for (int k = 0; k < 3; k++)
                for (int l = k; l < 3; l++)
                    sub_res += dot(V[k] - Pi, V[l] - Pi);
            res += 1.0 / 6.0 * tri_area * sub_res;
        }

        return res;
    }

    std::vector<Vector> vertices;
};

void save_frame(const std::vector<Polygon> &cells, std::string filename,
                int frameid = 0) {
    constexpr int W = 800, H = 800;
    constexpr double edge_width = 2.0;
    constexpr double edge_width2 = edge_width * edge_width;

    std::vector<unsigned char> inside(W * H, 0), edge(W * H, 0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)cells.size(); ++i) {
        const auto &V = cells[i].vertices;
        const int n = (int)V.size();
        if (n < 3)
            continue;

        std::vector<double> xs(n), ys(n);
        double xmin = 1e30, ymin = 1e30, xmax = -1e30, ymax = -1e30;
        for (int j = 0; j < n; ++j) {
            xs[j] = V[j][0] * W;
            ys[j] = V[j][1] * H;
            xmin = std::min(xmin, xs[j]);
            ymin = std::min(ymin, ys[j]);
            xmax = std::max(xmax, xs[j]);
            ymax = std::max(ymax, ys[j]);
        }

        int x0 = std::max(0, (int)std::floor(xmin - edge_width));
        int y0 = std::max(0, (int)std::floor(ymin - edge_width));
        int x1 = std::min(W - 1, (int)std::ceil(xmax + edge_width));
        int y1 = std::min(H - 1, (int)std::ceil(ymax + edge_width));
        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5, py = y + 0.5;

                int prev_sign = 0;
                bool isInside = true;
                bool isEdge = false;

                for (int j = 0; j < n; ++j) {
                    int k = (j + 1) % n;

                    double ax = xs[j], ay = ys[j];
                    double bx = xs[k], by = ys[k];
                    double dx = bx - ax, dy = by - ay;
                    double qx = px - ax, qy = py - ay;

                    double det = qx * dy - qy * dx;
                    int s = (det > 1e-12) - (det < -1e-12);

                    if (s != 0) {
                        if (prev_sign != 0 && s != prev_sign) {
                            isInside = false;
                            break;
                        }
                        prev_sign = s;
                    }

                    double len2 = dx * dx + dy * dy;
                    double dot = qx * dx + qy * dy;
                    if (dot >= 0.0 && dot <= len2 &&
                        det * det <= edge_width2 * len2)
                        isEdge = true;
                }

                if (isInside) {
                    int id = (H - 1 - y) * W + x;
                    inside[id] = 1;
                    if (isEdge)
                        edge[id] = 1;
                }
            }
        }
    }

    std::vector<unsigned char> image(W * H * 3, 255);

#pragma omp parallel for
    for (int i = 0; i < W * H; ++i) {
        if (edge[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 0;
        } else if (inside[i]) {
            image[3 * i + 0] = 0;
            image[3 * i + 1] = 0;
            image[3 * i + 2] = 255;
        }
    }

    std::ostringstream os;
    os << filename << frameid << ".png";
    stbi_write_png(os.str().c_str(), W, H, 3, image.data(), W * 3);
}

// saves a static svg file. The polygon vertices are supposed to be in the range
// [0..1], and a canvas of size 1000x1000 is created
void save_svg(const std::vector<Polygon> &polygons, std::string filename,
              const std::vector<Vector> *points = NULL,
              std::string fillcol = "none") {
    FILE *f = fopen(filename.c_str(), "w+");
    fprintf(f, "<svg xmlns = \"http://www.w3.org/2000/svg\" width = \"1000\" "
               "height = \"1000\">\n");
    for (int i = 0; i < polygons.size(); i++) {
        fprintf(f, "<g>\n");
        fprintf(f, "<polygon points = \"");
        for (int j = 0; j < polygons[i].vertices.size(); j++) {
            fprintf(f, "%3.3f, %3.3f ", (polygons[i].vertices[j][0] * 1000),
                    (1000 - polygons[i].vertices[j][1] * 1000));
        }
        fprintf(f, "\"\nfill = \"%s\" stroke = \"black\"/>\n", fillcol.c_str());
        fprintf(f, "</g>\n");
    }

    if (points) {
        fprintf(f, "<g>\n");
        for (int i = 0; i < points->size(); i++) {
            fprintf(f, "<circle cx = \"%3.3f\" cy = \"%3.3f\" r = \"3\" />\n",
                    (*points)[i][0] * 1000., 1000. - (*points)[i][1] * 1000);
        }
        fprintf(f, "</g>\n");
    }

    fprintf(f, "</svg>\n");
    fclose(f);
}

bool is_point_on_segment(const Vector &P, const Vector &A, const Vector &B) {
    Vector AP = P - A;
    Vector AB = B - A;
    double AP_dot_AB = dot(AP, AB);
    double AB_length_squared = dot(AB, AB);

    return AP_dot_AB > -eps && AP_dot_AB < AB_length_squared + eps;
}

Polygon make_regular_polygon(Vector center, double radius, int n) {
    // makes regular polygon of radius size radius with n vertices centered at
    // center
    Polygon result;
    for (int i = 0; i < n; i++) {
        result.vertices.push_back(
            center +
            radius * Vector(cos(2 * M_PI * i / n), sin(2 * M_PI * i / n)));
    }
    return result;
}

class VoronoiDiagram {

  public:
    VoronoiDiagram(){};

    void compute() {

        // For all sites Pi (in parallel) :
        //      Start with a unit square
        //      For all other sites Pj (optionally, only k nearest neighbors) :
        //          Clip it with bisector of [Pi,Pj]
        //      (fluids) : also clip it by a disk of radius sqrt(w_i -
        //      w_air) centered at Pi
        cells.clear();
        cells.resize(points.size());

        PointCloud<double> pc;
        pc.pts.resize(points.size());

        auto maxi = 2 * *std::max_element(weights.begin(), weights.end());
        maxi = std::max(maxi, eps); // negative weights

        for (size_t i = 0; i < points.size(); i++) {
            pc.pts[i].x = points[i][0];
            pc.pts[i].y = points[i][1];
            pc.pts[i].z = sqrt(maxi - weights[i]);
        }
        // straight from nanoFlann's demo (examples/pointcloud_kdd_radius.cpp)
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
            PointCloud<double>, 3>
            my_kd_tree;

        my_kd_tree index(3, pc, {10});
        index.buildIndex();

#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < points.size(); i++) {
            auto &Pi = points[i];
            Polygon result;
            result.vertices.push_back(Vector(0.0, 0.0));
            result.vertices.push_back(Vector(1.0, 0.0));
            result.vertices.push_back(Vector(1.0, 1.0));
            result.vertices.push_back(Vector(0.0, 1.0));

            size_t num_results = 151; // 150 + maybe one is our point ffs
            std::vector<uint32_t> ret_index(num_results);
            std::vector<double> out_dist_sqr(num_results);
            num_results = index.knnSearch(&Pi[0], num_results, &ret_index[0],
                                          &out_dist_sqr[0]);
            for (size_t j = 0; j < num_results; j++) {
                if (ret_index[j] == i)
                    continue;
                auto &Pj = points[ret_index[j]];
                result = clip_by_bisector(result, Pi, Pj, weights[i],
                                          weights[ret_index[j]]);
            }

            // clip it by regular poligon with 50 vertices
            if (weights[i] - weights.back() > eps) {
                double radius = sqrt(
                    weights[i] - weights.back()); // last elem is used for w_air
                Polygon regular = make_regular_polygon(Pi, radius, 50);
                for (size_t j = 0; j < regular.vertices.size(); j++) {
                    auto &A = regular.vertices[j];
                    auto &B =
                        regular.vertices[(j + 1) % regular.vertices.size()];
                    result = clip_by_edge(result, A, B);
                }
            } else {
                result.vertices.clear(); // particle vanishes
            }
            cells[i] = result;
        }
    }

    static Polygon clip_by_edge(const Polygon &V, const Vector &u,
                                const Vector &v) {

        // (fluids)
        // Clip a polygon by an edge defined by vertices u and v
        // Will be used to clip a polygon (a cell) by all the edges of a
        // (discretized) disk

        const Vector N =
            Vector(v.data[1] - u.data[1], -(v.data[0] - u.data[0]));

        Polygon result;
        result.vertices.clear(); // redundant
        for (size_t i = 0; i < V.vertices.size(); i++) {
            auto &A = V.vertices[i];
            auto &B = V.vertices[(i + 1) % V.vertices.size()];

            bool A_inside = dot(u - A, N) > -eps;
            bool B_inside = dot(u - B, N) > -eps;

            if (A_inside != B_inside) {
                auto P = A + dot(u - A, N) / dot(B - A, N) * (B - A);
                if (result.vertices.empty() ||
                    (result.vertices.back() - P).norm2() >
                        eps) { // prevents duplicate points in case the edge
                               // goes right through a vertex
                    result.vertices.push_back(P);
                }
            }
            if (B_inside) {
                if (result.vertices.empty() ||
                    (result.vertices.back() - B).norm2() > eps) {
                    result.vertices.push_back(B);
                }
            }
        }
        return result;
    }

    static Polygon clip_by_bisector(const Polygon &V, const Vector &P0,
                                    const Vector &Pi, double w0 = 0.0,
                                    double wi = 0.0) {

        // Lab 1 (Voronoi) : in Lab 1, we assume w0 = w1 = 0
        // Clip a polygon by the bisector of the segment defined by P0 (the
        // current site of the Voronoi cell being computed) and Pi (another
        // site)

        // Lab 2 (Semi-Discrete Optimal Transport) : extend to Laguerre
        // cells, i.e., w0 != w1

        const Vector M = (Pi + P0) / 2;
        const Vector M_prime =
            M + (w0 - wi) / (2 * dot(P0 - Pi, P0 - Pi)) * (Pi - P0);

        Polygon result;
        result.vertices.clear(); // redundant
        for (size_t i = 0; i < V.vertices.size(); i++) {
            auto &A = V.vertices[i];
            auto &B = V.vertices[(i + 1) % V.vertices.size()];

            bool A_inside =
                dot(A - P0, A - P0) - w0 < dot(A - Pi, A - Pi) - wi + eps;
            bool B_inside =
                dot(B - P0, B - P0) - w0 < dot(B - Pi, B - Pi) - wi + eps;

            if (A_inside != B_inside) {
                auto P = A + dot(M_prime - A, Pi - P0) / dot(B - A, Pi - P0) *
                                 (B - A);
                if (result.vertices.empty() ||
                    (result.vertices.back() - P).norm2() >
                        eps) { // prevents duplicate points in case the edge
                               // goes right through a vertex
                    result.vertices.push_back(P);
                }
            }
            if (B_inside) {
                if (result.vertices.empty() ||
                    (result.vertices.back() - B).norm2() > eps) {
                    result.vertices.push_back(B);
                }
            }
        }
        return result;
    }

    std::vector<Vector> points; // Lab 1 (Voronoi) : the sites to consider

    std::vector<double> weights; // Lab 2 (OT) : the weight associated to each
                                 // site (the Laguerre weight, i.e. the dual
                                 // optimal transport variables to be optimized)

    std::vector<Polygon>
        cells; // Lab 1 : the polygons representing each individual cell
};

// Lab 2
class OptimalTransport {

  public:
    OptimalTransport(){};

    void optimize(double fluid_volume);

    VoronoiDiagram vor;
    double fluid_volume;
};

// Labs 2 and 3
static lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *x,
                                lbfgsfloatval_t *g, const int n,
                                const lbfgsfloatval_t step) {
    OptimalTransport *ot = (OptimalTransport *)(instance);

    // first compute the Voronoi diagram at the current optimization step
    memcpy(&ot->vor.weights[0], x, n * sizeof(x[0]));
    ot->vor.compute();

    // (Optimal transport) : compute the function to be minimized
    // (fx) and its gradient (g[i], i=0..n-1)
    //
    // adapt these functions to support partial optimal transport (now "n" has
    // been increased by 1 to account for the air variable)

    lbfgsfloatval_t fx = 0.0;
    double target_area =
        ot->fluid_volume / ot->vor.points.size(); // all the lambda_i
    double estimated_vol_air = 1.0;

    for (int i = 0; i < n - 1; i++)
        estimated_vol_air -= ot->vor.cells[i].area();

    double quant = (1 - ot->fluid_volume) -
                   estimated_vol_air; // desired_vol_air - estimated_vol_air

    for (int i = 0; i < n - 1; i++) {
        fx -= ot->vor.cells[i].integral_square_distance(ot->vor.points[i]) -
              ot->vor.cells[i].area() * ot->vor.weights[i] +
              target_area * ot->vor.weights[i];
        g[i] = -(target_area - ot->vor.cells[i].area());
    }
    fx -= ot->vor.weights.back() * quant;
    g[n - 1] = -quant;

    return fx;
}

// Labs 2 and 3 : you may use this function to print debugging info.
static int progress(void *instance, const lbfgsfloatval_t *x,
                    const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                    const lbfgsfloatval_t step, int n, int k, int ls) {
    printf("Iteration %d:\n", k);
    printf("  fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

// Lab 2
void OptimalTransport::optimize(double fluid_volume = 1.0) {

    lbfgsfloatval_t fx;
    std::vector<double> weights(vor.weights);

    this->fluid_volume = fluid_volume;

    lbfgs_parameter_t param;
    // Initialize the parameters for the L-BFGS optimization.
    lbfgs_parameter_init(&param);

    // run the LBFGS optimizer
    int ret = lbfgs(weights.size(), &weights[0], &fx, evaluate, progress,
                    (void *)this, &param);

    // copy the result back to the voronoi structure
    vor.weights = weights;

    // finally recompute the Voronoi diagram with the final optimized weights
    vor.compute();
}

class Fluid {
  public:
    Fluid(int N_particles = 1000) : N_particles(N_particles) {}

    void compute_vor() {
        for (int i = 0; i < N_particles; i++)
            ot.vor.points[i] = particles[i];
        ot.vor.compute();
        ot.optimize(fluid_volume);
    }

    // advance the simulation dt in time
    void time_step(double dt) {

        double epsilon2 = 0.004 * 0.004;
        Vector g(0, -9.81);
        double m_i = 200;

        // Compute semi-discrete partial optimal transport
        // for all particles, add gravity and spring force towards cell
        // centroid, integrate acceleration->velocity and velocity->position
        std::vector<Vector> new_particles(N_particles),
            new_velocities(N_particles);
        for (int i = 0; i < N_particles; i++) {
            Vector F_i_spring(0, 0);
            if (ot.vor.cells[i].area() > eps) {
                F_i_spring = 1.0 / epsilon2 *
                             (ot.vor.cells[i].centroid() - particles[i]);
            }
            Vector F_i = F_i_spring + m_i * g;
            new_velocities[i] = velocities[i] + dt / m_i * F_i;
            new_particles[i] = particles[i] + dt * new_velocities[i];
            // new_velocities[i] = new_velocities[i] * 0.99; // air friction
            double bounce = 0.0; // full damping
            if (new_particles[i][0] < eps) {
                new_particles[i][0] = eps;
                new_velocities[i][0] = std::abs(new_velocities[i][0]) * bounce;
            } else if (new_particles[i][0] > 1.0 - eps) {
                new_particles[i][0] = 1.0 - eps;
                new_velocities[i][0] = -std::abs(new_velocities[i][0]) * bounce;
            }

            if (new_particles[i][1] < eps) {
                new_particles[i][1] = eps;
                new_velocities[i][1] = std::abs(new_velocities[i][1]) * bounce;
            } else if (new_particles[i][1] > 1.0 - eps) {
                new_particles[i][1] = 1.0 - eps;
                new_velocities[i][1] = -std::abs(new_velocities[i][1]) * bounce;
            }
        }
        particles = new_particles;
        velocities = new_velocities;
        compute_vor();
    }

    // just run the full simulation
    void run_simulation() {
        compute_vor();
        double dt = 0.01;
        std::filesystem::create_directory("fluid_video");
        for (int i = 0; i < 1000; i++) {
            time_step(dt);
            save_frame(ot.vor.cells, "fluid_video/frame_", i);
        }
    }

    int N_particles;

    OptimalTransport ot;
    std::vector<Vector> particles;  // the position of all particles
    std::vector<Vector> velocities; // the velocities of all particles
    double fluid_volume; // you decide the fraction of the unit square occupied
                         // by the fluid
};

static std::default_random_engine engine;
thread_local std::uniform_real_distribution<double> uniform(0, 1);

int main() {

    Fluid fluid(700);

    engine.seed(0);

    double radius = 0.3;
    fluid.fluid_volume = M_PI * radius * radius;

    for (int i = 0; i < fluid.N_particles; i++) {
        double r = radius * sqrt(uniform(engine));
        double theta = 2 * M_PI * uniform(engine);
        double x = 0.5 + r * cos(theta);
        double y = 0.55 + r * sin(theta);
        fluid.particles.push_back(Vector(x, y));
        fluid.ot.vor.points.push_back(Vector(x, y));
        fluid.velocities.push_back(Vector(0, 0));
    }
    fluid.ot.vor.weights.resize(fluid.N_particles + 1, 0.0);

    fluid.run_simulation();

    return 0;
}
