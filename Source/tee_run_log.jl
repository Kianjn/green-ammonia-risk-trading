# ==============================================================================
# tee_run_log.jl — Save the post-run console summary into the results folder
# ==============================================================================
#
# After ADMM converges / hits max_iter (or the social planner finishes), the
# save_* routines print cost metrics and the run summary. Wrap those prints
# with with_run_summary_log so the same text is written to run_summary.txt.
#
# Julia 1.12 `redirect_stdout` only accepts Pipe / DevNull / IOStream — not a
# custom TeeStream. Capture into a real temp file, then replay to the terminal
# and to run_summary.txt. If redirect fails, still print to the console.
#
# ==============================================================================

using Dates
using Logging

function with_run_summary_log(f::Function, results_dir::AbstractString;
                              filename::String = "run_summary.txt")
    isdir(results_dir) || mkpath(results_dir)
    log_path = joinpath(results_dir, filename)
    header = sprint() do io
        script = !isempty(PROGRAM_FILE) ? PROGRAM_FILE : "repl"
        println(io, "# run_summary.txt")
        println(io, "# script:  ", script)
        println(io, "# written: ", Dates.now())
        println(io, "# results: ", results_dir)
        println(io)
    end

    orig_logger = global_logger()
    tmp_path = tempname()
    text = ""
    try
        open(tmp_path, "w") do io
            redirect_stdout(io) do
                redirect_stderr(io) do
                    global_logger(ConsoleLogger(io))
                    Base.invokelatest(f)
                end
            end
        end
        text = isfile(tmp_path) ? read(tmp_path, String) : ""
        print(stdout, text)
        flush(stdout)
        open(log_path, "w") do log
            print(log, header)
            print(log, text)
        end
    catch err
        @warn "Could not tee run summary to file; printing to console only" exception=(err, catch_backtrace())
        global_logger(orig_logger)
        try
            Base.invokelatest(f)
        catch
            rethrow()
        end
        try
            open(log_path, "w") do log
                print(log, header)
                println(log, "# (console tee failed: ", typeof(err), ")")
            end
        catch
        end
    finally
        global_logger(orig_logger)
        isfile(tmp_path) && rm(tmp_path; force=true)
    end
    return log_path
end
