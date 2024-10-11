module TopModule (
    input        clk,
    input        reset,
    input        data,
    input        ack,
    output [3:0] count,
    output       counting,
    output       done
);

    // Internal signals
    wire pattern_detected;
    wire [3:0] delay;
    wire start_count;
    wire timer_done;
    wire timer_counting;
    wire [3:0] remaining_time;
    wire reset_search;

    // Instantiate PatternDetector
    PatternDetector PatternDetector_inst (
        .clk(clk),
        .reset(reset | reset_search),
        .data(data),
        .pattern_detected(pattern_detected)
    );

    // Instantiate ShiftRegister
    ShiftRegister ShiftRegister_inst (
        .clk(clk),
        .reset(reset | reset_search),
        .shift_enable(pattern_detected),
        .data_in(data),
        .data_out(delay)
    );

    // Instantiate TimerCounter
    TimerCounter TimerCounter_inst (
        .clk(clk),
        .reset(reset | reset_search),
        .start_count(pattern_detected),
        .delay(delay),
        .count(remaining_time),
        .counting(counting),
        .done(timer_done),
        .remaining_time(remaining_time)
    );

    // Instantiate AckController
    AckController AckController_inst (
        .clk(clk),
        .reset(reset),
        .done(timer_done),
        .ack(ack),
        .reset_search(reset_search)
    );

    // Connect TimerCounter outputs to TopModule outputs
    assign count = remaining_time;
    assign done  = timer_done;

endmodule