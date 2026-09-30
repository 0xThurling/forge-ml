return {
    project = {
        name = "ml",
        type = "executable",
        standard = "23",
    },
    testing = {
        enabled = true,
        framework = "gtest",
        benchmark = true,
    },
    dependencies = {
        direct = {
            forgefp = {
                path = "../fp",
                target = "forgefp",
            },
            googletest = {
                git = "https://github.com/google/googletest.git",
                tag = "v1.14.0",
            },
        },
        conan = {},
    },
    build = {
        presets = { "warnings", "concurrency" },
    },
    resources = {
        files = {},
    },
    scripts = {},
    features = {
    },
}
