
% Disjunction of Conjunction form - canonical normal form
% % Rule for c0 direction %%
predict_direction(A0, A2, A3, _,A5, A6, A8, A10, A13, c0) :-
    (   A3 = false, A10 = false ;
        A3 = false, A10 = true, A2 = true ;
        A3 = false, A10 = true, A2 = false, A13 = true, A0 = false ;
        A3 = true, A0 = false, A8 = true, A10 = false;
        A3 = true, A0 = true, A10 = false, A8 = true;
        A3 = true, A0 = true, A10 = true, A5 = true, A6 = false
    ).

% Rule for c1 direction
predict_direction(A0, _, A3, _, A5, _, A8, A10, _, c1) :-
    (   A3 = true, A0 = false, A8 = false ;
        A3 = true, A0 = false, A8 = true, A10 = true ;
        A3 = true, A0 = true, A10 = true, A5 = false
    ).

% Rule for c2 direction
predict_direction(A0, _, A3, A4, _, _, A8, A10, _, c2) :-
    (  A3 = true, A0 = true, A10 = false, A8 = false, A4 = false
    ).

% Rule for c3 direction
predict_direction(A0, A2, A3, A4, A5, A6, A8, A10, A13, c3) :-
    (   A3 = false, A10 = true, A2 = false, A13 = false;
        A3 = false, A10 = true, A2 = false, A13 = true, A0 = true;
        A3 = true, A0 = true, A10 = false, A8 = false, A4 = true;
        A3 = true, A0 = true, A10 = true, A5 = true, A6 = true
    ).

% Query with user prompts
predict_direction_on_user_input :-
    write('Enter the value for A0 (true/false): '), read(A0),
    write('Enter the value for A2 (true/false): '), read(A2),
    write('Enter the value for A3 (true/false): '), read(A3),
    write('Enter the value for A4 (true/false): '), read(A4),
    write('Enter the value for A5 (true/false): '), read(A5),
    write('Enter the value for A6 (true/false): '), read(A6),
    write('Enter the value for A8 (true/false): '), read(A8),
    write('Enter the value for A10 (true/false): '), read(A10),
    write('Enter the value for A13 (true/false): '), read(A13),
    predict_direction(A0, A2, A3, A4, A5, A6, A8, A10, A13, Direction),
    format('Predicted direction: ~w', [Direction]).





